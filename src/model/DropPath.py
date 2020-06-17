import tensorflow as tf

class DropPathTracker(tf.keras.callbacks.Callback):
    def __init__(self, base_drop_path_chance: float, epochs_so_far: int, total_epochs: int, steps_multiplier: float = 1.):
        super().__init__()
        self.drop_path_chance = tf.Variable(1., trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False, dtype=tf.int32)
        self.total_steps = tf.Variable(0, dtype=tf.int32)
        self.global_step.assign(0)

        self.base_drop_path_chance = base_drop_path_chance
        self.drop_path_chance.assign(1.0)

        self.epochs_so_far = epochs_so_far
        self.total_epochs = total_epochs
        self.steps_multiplier = steps_multiplier


    def on_train_begin(self, logs=None):
        self.drop_path_chance.assign(self.base_drop_path_chance)
        steps_per_epoch = self.params['steps']
        steps_so_far = self.epochs_so_far * steps_per_epoch
        total_steps = int(self.total_epochs * steps_per_epoch * self.steps_multiplier)
        self.global_step.assign(steps_so_far)
        self.total_steps.assign(total_steps)


    def on_train_end(self, logs=None):
        self.drop_path_chance.assign(1.0)

    def on_batch_end(self, batch, logs=None):
        self.global_step.assign_add(1)


class DropPathOperation(tf.keras.layers.Layer):
    def __init__(self, cell_position_as_ratio: float, drop_path_tracker: DropPathTracker = None, **kwargs):
        super().__init__(**kwargs)

        self.drop_path_tracker = drop_path_tracker
        self.cell_position_as_ratio = cell_position_as_ratio

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'cell_position_as_ratio': self.cell_position_as_ratio
        })
        return config

    @tf.function
    def calculate_drop_path(self, inputs, chance):
        layer = inputs

        batch_size = tf.shape(input=inputs)[0]
        noise_shape = [batch_size, 1, 1, 1]
        random_tensor = chance
        random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
        binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
        keep_prob_inv = tf.cast(1.0 / chance, inputs.dtype)
        layer = layer * keep_prob_inv * binary_tensor

        return layer

    @tf.function
    def scale_chance_by_cell(self, chance):
        return 1 - self.cell_position_as_ratio * (1 - chance)

    @tf.function
    def scale_chance_by_steps(self, chance):
        ratio = tf.divide(self.drop_path_tracker.global_step, self.drop_path_tracker.total_steps)
        ratio = tf.cast(ratio, tf.float32)
        ratio = tf.minimum(1.0, ratio)
        return 1 - ratio * (1 - chance)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        layer = inputs
        if training:
            base_chance = self.drop_path_tracker.drop_path_chance if self.drop_path_tracker is not None else 1.

            if base_chance < 1.0:
                chance = self.scale_chance_by_cell(base_chance)
                chance = self.scale_chance_by_steps(chance)
                layer = self.calculate_drop_path(layer, chance)
            # layer = tf.cond(base_chance < 1.0, layer, )

        return layer