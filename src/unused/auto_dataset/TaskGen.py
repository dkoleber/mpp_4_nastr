from __future__ import annotations
from enum import Enum, IntEnum

from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import math
from typing import List, Tuple
import os
import cv2

from Dataset import ImageDataset
from FileManagement import *
from Utils import *

def place(orig_img, new_img, coord) -> None:
    new_size = (new_img.size[0] + coord[0], new_img.size[1] + coord[1])
    shifted_image = new_img.transform(new_size, Image.AFFINE, (1, 0, -coord[0], 0, 1, -coord[1]))
    orig_img.alpha_composite(shifted_image)


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
    return (get_random_int(360, 0), 100, 100) #hsv rather than rgb


class ObjectModifier(IntEnum):
    SizeModifier = 1,
    PerspectiveModifier = 2,
    RotationModifier = 3,
    ColorModifier = 4


def _name_from_modifier(mod: ObjectModifier) -> str:
    if mod == ObjectModifier.SizeModifier:
        return 'size'
    if mod == ObjectModifier.RotationModifier:
        return 'rotation'
    if mod == ObjectModifier.ColorModifier:
        return 'color'
    else:
        return 'perspective'


class ObjectRep:
    def __init__(self, num_verticies: int, color: Tuple = None):
        self.verticies = np.random.random((num_verticies, 2))
        self.verticies = list(self.verticies)
        self.verticies = [list(x) for x in self.verticies]

        self.color = color #REPRESENTED AS HSV RATHER THAN RGB
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

    def _draw(self, scale: int, modifiers: List[ObjectModifier] = None):
        width = math.ceil(self.width * scale)
        height = math.ceil(self.height * scale)
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        mods = [] if modifiers is None else modifiers

        color_to_use = [x for x in self.color]

        if ObjectModifier.ColorModifier in mods:
            shift = 15
            color_shift = get_random_int(shift,-shift)
            color_to_use[0] += color_shift
            color_to_use[0] %= 360

        # scaled = [((v[0] + self.top_left_corner[0]) * scale_to_use + position[0], (v[1] + self.top_left_corner[1]) * scale_to_use + position[1]) for v in self.verticies]
        scaled = [(math.ceil((v[0]- self.top_left_corner[0]) * scale) , math.ceil((v[1] - self.top_left_corner[1]) * scale)) for v in self.verticies]

        draw.polygon(scaled, fill=f'hsv({color_to_use[0]}, {color_to_use[1]}%, {color_to_use[2]}%)')

        for mod in mods:
            if mod == ObjectModifier.SizeModifier:
                max_diff = .5

                resize_factor = (1 - max_diff) + (np.random.random() * 2 * max_diff)
                new_size = (math.ceil(width * resize_factor), math.ceil(height * resize_factor))
                img = img.resize(new_size)

            if mod == ObjectModifier.PerspectiveModifier:
                ratio = .9

                y_coords_top = (np.random.random(2) * height * (1. - ratio))
                x_coords_left = (np.random.random(2) * width * (1. - ratio))
                y_coords_bottom = (np.random.random(2) * height * (1. - ratio)) + (ratio * height)
                x_coords_right = (np.random.random(2) * width * (1. - ratio)) + (ratio * width)

                from_coords = [(0,0), (width, 0), (width, height), (0, height)]
                to_coords = [(x_coords_left[0], y_coords_top[0]), (x_coords_right[1], y_coords_top[1]),
                             (x_coords_right[1], y_coords_bottom[1]), (x_coords_left[1], y_coords_bottom[0])]

                coeffs = find_coeffs(to_coords, from_coords)
                img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
                #TODO: realign with top left corner

            if mod == ObjectModifier.RotationModifier:
                max_rotation = 15
                rotation_amount = get_random_int(max_rotation, -max_rotation)
                img = img.rotate(rotation_amount, Image.NEAREST, expand = 1)
                # TODO: realign with top left corner

        return img

    def draw(self, scale: int, modifiers: List[ObjectModifier] = None):
        mods = [] if modifiers is None else modifiers

        color_to_use = [x for x in self.color]

        def get_zero_aligned_vertices(verts):
            min_x = verts[0][0]
            min_y = verts[0][1]

            for v in verts:
                if v[0] < min_x:
                    min_x = v[0]
                if v[1] < min_y:
                    min_y = v[1]

            res = [(x[0] - min_x, x[1] - min_y) for x in verts]

            max_x = 0
            max_y = 0

            for v in res:
                if v[0] > max_x:
                    max_x = v[0]
                if v[1] > max_y:
                    max_y = v[1]

            return res, max_x, max_y

        modified_vertices, height, width = get_zero_aligned_vertices(self.verticies)

        for mod in mods:
            if mod == ObjectModifier.SizeModifier:
                max_diff = .5
                resize_factor = (1 - max_diff) + (np.random.random() * 2 * max_diff)
                modified_vertices = [(x[0] * resize_factor, x[1] * resize_factor) for x in modified_vertices]
                height *= resize_factor
                width *= resize_factor

            if mod == ObjectModifier.PerspectiveModifier:
                ratio = .9

                y_coords_top = (np.random.random(2) * height * (1. - ratio))
                x_coords_left = (np.random.random(2) * width * (1. - ratio))
                y_coords_bottom = (np.random.random(2) * height * (1. - ratio)) + (ratio * height)
                x_coords_right = (np.random.random(2) * width * (1. - ratio)) + (ratio * width)

                from_coords = np.array([(0,0), (width, 0), (width, height), (0, height)], dtype=np.float32)
                to_coords = np.array([(x_coords_left[0], y_coords_top[0]), (x_coords_right[1], y_coords_top[1]),
                             (x_coords_right[1], y_coords_bottom[1]), (x_coords_left[1], y_coords_bottom[0])], dtype=np.float32)

                transformation_matrix = cv2.getPerspectiveTransform(from_coords, to_coords)
                transformed_coords = cv2.perspectiveTransform(np.array(modified_vertices, dtype=np.float32)[np.newaxis], transformation_matrix)[0]
                modified_vertices, height, width = get_zero_aligned_vertices(transformed_coords)

            if mod == ObjectModifier.RotationModifier:
                max_rotation = 15
                rotation_amount = get_random_int(max_rotation, -max_rotation)
                rotation_radians = rotation_amount * math.pi / 180.

                rotated_vertices = []

                for v in modified_vertices:
                    angle = math.atan2(v[1], v[0])
                    magnitude = math.sqrt(v[1]** 2 + v[0]**2)
                    rotated_x = magnitude * math.cos(angle + rotation_radians)
                    rotated_y = magnitude * math.sin(angle + rotation_radians)
                    rotated_vertices.append((rotated_x, rotated_y))

                modified_vertices, height, width = get_zero_aligned_vertices(rotated_vertices)

            if mod == ObjectModifier.ColorModifier:
                shift = 15
                color_shift = get_random_int(shift,-shift)
                color_to_use[0] += color_shift
                color_to_use[0] %= 360


        # scaled = [((v[0] + self.top_left_corner[0]) * scale_to_use + position[0], (v[1] + self.top_left_corner[1]) * scale_to_use + position[1]) for v in self.verticies]
        # scaled = [(math.ceil((v[0]- self.top_left_corner[0]) * scale) , math.ceil((v[1] - self.top_left_corner[1]) * scale)) for v in self.verticies]
        scaled = [(math.ceil(x[0] * scale), math.ceil(x[1] * scale)) for x in modified_vertices]
        width *= scale
        height *= scale

        width = math.ceil(width)
        height = math.ceil(height)

        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.polygon(scaled, fill=f'hsv({color_to_use[0]}, {color_to_use[1]}%, {color_to_use[2]}%)')

        return img


class DatasetGenerator(ImageDataset):
    def __init__(self, mixed_train_images: np.ndarray, mixed_train_labels: np.ndarray, test_set_names: List[str], test_set_images: List[np.ndarray], test_set_labels: List[np.ndarray]):
        super().__init__(mixed_train_images, mixed_train_labels, 1., 0.)
        self.test_set_images = test_set_images
        self.test_set_labels = test_set_labels
        self.test_set_names = test_set_names

    @staticmethod
    def _build_task_dataset(dataset_size: int, image_size: Tuple, class_objs: List[ObjectRep], aux_objs: List[ObjectRep], object_schedule: List[int], objects_per_image:int, max_depth_of_target: int, modifiers:List[ObjectModifier]= None):
        num_classes = len(class_objs)


        def place_randomly(img: Image, obj: Image):
            x_coord = get_random_int(img.size[0] - obj.size[0])
            y_coord = get_random_int(img.size[1] - obj.size[1])
            place(img, obj, (x_coord, y_coord))

        scalar = int(image_size[0] / 2) #TODO: as parameter?
        # scalar = image_size[0]

        images = []

        for i in range(dataset_size):
            target_depth = min(get_random_int(max_depth_of_target), objects_per_image - 1)
            target_index = objects_per_image - target_depth - 1
            img = Image.new('RGBA', image_size, (255, 255, 255, 0))

            for o in range(objects_per_image):

                if o == target_index:
                    place_randomly(img, class_objs[object_schedule[i]].draw(scalar, modifiers))

                else:
                    place_randomly(img, aux_objs[get_random_int(len(aux_objs))].draw(scalar, modifiers))

            array_to_save = np.array(img)
            array_to_save = np.delete(array_to_save, (3), axis=2)  # remove alpha channel

            images.append(array_to_save)

        array_to_save = np.array(images)

        return array_to_save

    @staticmethod
    def build_task_dataset(dataset_size:int, image_size:Tuple, num_classes: int,
                 verticies_per_object:int,
                 objects_per_image:int,
                 save_location,
                 max_depth_of_target:int = 0,
                 modifiers:List[ObjectModifier]= None,
                 randomize_vertices: bool = False):
        max_verticies_per_object = max(3, verticies_per_object)

        core_classes = []
        aux_objects = []

        if randomize_vertices:
            core_classes = [ObjectRep(get_random_int(max_verticies_per_object, 3), (int(x * 360 / num_classes), 100, 100)) for x in range(num_classes)]
            aux_objects = [ObjectRep(get_random_int(max_verticies_per_object, 3), get_random_color()) for _ in range(int(dataset_size / 2))]
        else:
            core_classes = [ObjectRep(max_verticies_per_object, (int(x * 360 / num_classes), 100, 100)) for x in range(num_classes)]
            aux_objects = [ObjectRep(max_verticies_per_object, get_random_color()) for _ in range(int(dataset_size / 2))]

        repeats = int((dataset_size / num_classes) + 1)
        object_schedule = ([x for x in range(num_classes)] * repeats)[:dataset_size]

        full_dataset = DatasetGenerator._build_task_dataset(dataset_size, image_size, core_classes, aux_objects, object_schedule, objects_per_image, max_depth_of_target, modifiers)
        np.save(os.path.join(save_location, 'dataset.npy'), full_dataset)

        partial_set_size = int(dataset_size / 10)
        test_set_mods = [[x] for x in modifiers]
        test_set_mods.append([])

        for mod in test_set_mods:
            partial_dataset = DatasetGenerator._build_task_dataset(partial_set_size, image_size, core_classes, aux_objects, object_schedule[:partial_set_size], objects_per_image, max_depth_of_target, mod)
            mod_name = _name_from_modifier(mod[0]) if len(mod) == 1 else 'default'
            np.save(os.path.join(save_location, f'test_set_{mod_name}.npy'), partial_dataset)

        class_images = []

        for cl in core_classes:
            img = Image.new('RGBA', image_size, (255, 255, 255, 255))

            place(img, cl.draw(image_size[0]), (0,0))
            class_images.append(np.array(img))

        classes_to_save = np.array(class_images)

        if not os.path.exists(save_location):
            os.makedirs(save_location)


        np.save(os.path.join(save_location, 'labels.npy'), object_schedule)
        np.save(os.path.join(save_location, 'classes.npy'), classes_to_save)

    @staticmethod
    def get_task_dataset(dir_name) -> DatasetGenerator:
        if os.path.exists(dir_name):
            images = np.load(os.path.join(dir_name, 'dataset.npy'))
            labels = np.load(os.path.join(dir_name, 'labels.npy'))

            test_set_names = [x for x in os.listdir(dir_name) if 'test_set' in x]
            test_sets = [np.load(os.path.join(dir_name, x)) for x in test_set_names]
            test_set_labels = [labels[:len(x)] for x in test_sets]
            test_set_titles =  [x[9:] for x in test_set_names] #'test_set_'

            labels = np.reshape(labels, (labels.shape[0], 1))
            return DatasetGenerator(images, labels, test_set_titles, test_sets, test_set_labels)
        return DatasetGenerator([], [], [], [], [])

    @staticmethod
    def _dataset_name(mods: List[ObjectModifier]):
        mods.sort()
        mod_list_as_str = ''.join(['_' + _name_from_modifier(x) for x in mods])
        dataset_name = f'dataset{mod_list_as_str}'
        return dataset_name


def visualize_images(imgs, scale: int = 1):
    side_size = math.ceil(math.sqrt(len(imgs)))
    x_dim = len(imgs[0]) * scale
    y_dim = len(imgs[0][0]) * scale

    img = Image.new('RGBA', (side_size * x_dim, side_size * y_dim), (255, 255, 255, 128))

    for i, im in enumerate(imgs):
        x = x_dim * (i % side_size)
        y = y_dim * int(i / side_size)
        box = (x, y, x+x_dim, y+y_dim)
        resized = Image.fromarray(im).resize((x_dim, y_dim))
        img.paste(resized, box)

    img.show()

def test_dataset_generator():
    generator = DatasetGenerator.build_task_dataset(1000, (32, 32), 10, 4, 5, os.path.join(evo_dir, 'dataset_gen', ))

def test_show_classes():
    # classes = np.load(os.path.join(evo_dir, 'dataset_gen', 'classes.npy'))
    classes = np.load(os.path.join(evo_dir, 'dataset_gen', 'dataset.npy'))

    scale = 10

    for cl in classes[:4]:
        img = Image.fromarray(cl)
        img = img.resize((img.size[0] * scale, img.size[1] * scale))
        img.show()


if __name__ == '__main__':
    # test_object_rep()
    # test_transform()
    test_dataset_generator()
    test_show_classes()
    # test_overlays()