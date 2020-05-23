from enum import Enum

from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import math
from typing import List, Tuple
import os

from FileManagement import *

'''
each dataset has a set of attributes:
- image size
- number of classes
- object modifiers:
    - distortion/perspective shifting
    - size alteration
- number of non-class objects
    - depth of class image among other objects


each image has N classes

each image set has X possible 

'''

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    # A = np.array(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)



def get_random_color() -> Tuple:
    return tuple((np.random.random(3)*255).astype(int))

def get_random_int(max_val:int, min_val: int = 0):
    return int((np.random.random() * (max_val - min_val)) + min_val)

class ObjectModifier(Enum):
    SizeModifier = 1,
    PerspectiveModifier = 2,
    RotationModifier = 3,
    ColorModifier = 4

class ObjectRep:
    def __init__(self, num_verticies: int, color: Tuple = None):
        self.verticies = np.random.random((num_verticies, 2))
        self.verticies = list(self.verticies)
        self.verticies = [list(x) for x in self.verticies]

        self.color = color
        if self.color is None:
            self.color = get_random_color()

        # center_of_mass = list(np.random.random(2))

        self.center_of_mass = [0, 0]
        for v in self.verticies:
            self.center_of_mass[0] += v[0]
            self.center_of_mass[1] += v[1]

        #note: what is called COM here, is not actually center of mass in terms of area

        self.center_of_mass[0] /= float(num_verticies)
        self.center_of_mass[1] /= float(num_verticies)

        self.top_left_corner = [x for x in self.verticies[0]]
        self.bottom_right_corner = [x for x in self.verticies[0]]

        for v in self.verticies:
            for i in range(2):
                if v[i] < self.top_left_corner[i]:
                    self.top_left_corner[i] = v[i]
                if v[i] > self.bottom_right_corner[i]:
                    self.bottom_right_corner[i] = v[i]

        self.width = self.bottom_right_corner[0] - self.top_left_corner[0]
        self.height = self.bottom_right_corner[1] - self.top_left_corner[1]

        def coord_as_angle(coord):
            # nonlocal center_of_mass
            delta_x = self.center_of_mass[0] - coord[0]
            delta_y = self.center_of_mass[1] - coord[1]

            return math.atan2(delta_y, delta_x)

        self.verticies.sort(key=coord_as_angle)

    def __repr__(self):
        return str(self.verticies)

    def draw(self, scale: int, modifiers: List[ObjectModifier] = None):
        width = int(self.width * scale)
        height = int(self.height * scale)
        img = Image.new('RGBA', (height, width), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        mods = [] if modifiers is None else modifiers

        # scaled = [((v[0] + self.top_left_corner[0]) * scale_to_use + position[0], (v[1] + self.top_left_corner[1]) * scale_to_use + position[1]) for v in self.verticies]
        scaled = [(int(v[0] * scale), int(v[1] * scale)) for v in self.verticies]




        draw.polygon(scaled, fill=self.color)

        for mod in mods:
            if mod == ObjectModifier.SizeModifier:
                resize_factor = .5 + np.random.random()
                img = img.resize((int(width * resize_factor), int(height * resize_factor)))

            if mod == ObjectModifier.PerspectiveModifier:
                x_coords_top = np.sort(np.random.random(2) * width)
                y_coords_left = np.sort(np.random.random(2) * height)
                x_coords_bottom = np.sort(np.random.random(2) * width)
                y_coords_right = np.sort(np.random.random(2) * height)

                from_coords = [(0,0), (width, 0), (width, height), (0, height)]
                to_coords = [(x_coords_top[0], y_coords_left[0]), (x_coords_top[1], y_coords_right[0]),
                             (x_coords_bottom[1], y_coords_right[1]), (x_coords_bottom[0], y_coords_left[1])]

                coeffs = find_coeffs(to_coords, from_coords)
                img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

            if mod == ObjectModifier.RotationModifier:
                rotation_amount = get_random_int(45, -45)
                img = img.rotate(rotation_amount, Image.NEAREST)

            if mod == ObjectModifier.ColorModifier:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(5. + np.random.random())

        return img

class DatasetGenerator:
    def __init__(self, dataset_size:int, image_size:Tuple, num_classes: int,
                 max_verticies_per_object:int,
                 objects_per_image:int,
                 save_location,
                 max_depth_of_target:int = 0,
                 modifiers:List[ObjectModifier]= None):
        self.max_verticies_per_object = max(3, max_verticies_per_object)


        core_classes = [ObjectRep(get_random_int(self.max_verticies_per_object, 3), get_random_color()) for _ in range(num_classes)]

        aux_objects = [ObjectRep(get_random_int(self.max_verticies_per_object, 3), get_random_color()) for _ in range(int(dataset_size / 2))]

        self.images = []

        repeats = int((dataset_size/num_classes)+1)
        object_schedule = ([x for x in range(num_classes)]*repeats)[:dataset_size]

        def random_coordinates():
            nonlocal image_size
            return (np.random.random()*image_size[0], np.random.random()*image_size[1])

        scalar = int(image_size[0] / 2)
        for i in range(dataset_size):
            target_depth = get_random_int(max_depth_of_target)
            img = Image.new('RGBA', image_size, (255, 255, 255, 0))
            img_draw = ImageDraw.Draw(img)

            for o in range(objects_per_image):


                if o == objects_per_image-(1+target_depth):
                    core_classes[object_schedule[i]].draw(img_draw, scalar, random_coordinates())
                else:
                    aux_objects[get_random_int(len(aux_objects))].draw(img_draw, scalar, random_coordinates())

            self.images.append(np.array(img))

        array_to_save = np.array(self.images)

        self.class_images = []

        for cl in core_classes:
            img = Image.new('RGBA', image_size, (255, 255, 255, 0))
            img_draw = ImageDraw.Draw(img)

            cl.draw(img_draw, image_size[0], (int(image_size[0]/2), int(image_size[0]/2)))
            self.class_images.append(np.array(img))

        classes_to_save = np.array(self.class_images)

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        np.save(os.path.join(save_location, 'dataset.npy'), array_to_save)
        np.save(os.path.join(save_location, 'classes.npy'), classes_to_save)


def test_object_rep():
    img = Image.new('RGBA', (500, 500), (255, 255, 255, 0))

    draw = ImageDraw.Draw(img)

    for i in range(200):
        obj = ObjectRep(6)
        obj.draw(draw, 250, tuple(np.random.random(2) * 500))

        # print(obj)

    img.show()

def test_dataset_generator():
    generator = DatasetGenerator(1000, (32,32), 10, 4, 10, os.path.join(evo_dir, 'dataset_gen', ))

def test_show_first_image():
    classes = np.load(os.path.join(evo_dir, 'dataset_gen', 'classes.npy'))
    print(classes.shape)

    img = Image.fromarray(classes[0])

    img.show()


def test_transform():
    dim = 500

    img = Image.new('RGBA', (dim, dim), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    obj = ObjectRep(6)
    obj.draw(draw, dim, tuple(np.random.random(2) * 500))

    img.show()

    coeffs = find_coeffs(
                        [(0, 0), (dim / 2, 0), (dim, dim), (0, dim)], #to
                        [(0, 0), (dim, 0), (dim, dim), (0, dim)]) #from
    img = img.transform((dim,dim), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    img.show()

def test_overlays():
    dim = 500
    img = Image.new('RGBA', (dim, dim), (0, 0, 0, 0))

    obj1 = ObjectRep(6)
    obj2 = ObjectRep(6)

    def place(orig_img, new_img, coord):
        shifted_image = new_img.transform(new_img.size, Image.AFFINE, (1, 0, coord[0], 0, 1, coord[1]))
        orig_img.alpha_composite(new_img)


    place(img, obj1.draw(dim), (0,0))
    place(img, obj2.draw(dim), (0,0))
    # img = Image.alpha_composite(img, obj2.draw(dim))

    img.show("unmodified")

    mods = [
        ObjectModifier.SizeModifier,
        # ObjectModifier.ColorModifier,
        # ObjectModifier.RotationModifier,
        # ObjectModifier.PerspectiveModifier
    ]

    img2 = Image.new('RGBA', (dim, dim), (0, 0, 0, 0))

    place(img2, obj1.draw(dim, mods), (0, 0))
    place(img2, obj2.draw(dim, mods), (0, 0))

    img2.show("modified")


if __name__ == '__main__':
    # test_object_rep()
    # test_transform()
    # test_dataset_generator()
    # test_show_first_image()
    test_overlays()