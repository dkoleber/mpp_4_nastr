import  tensorflow as tf
import numpy as np

class SGDR(tf.keras.callbacks.Callback):
    def __init__(self, minimum_learning_rate, maximum_learning_rate, batch_size, epoch_size, epochs_per_restart: float = .25, learning_rate_decay: float = 1., restart_period_decay: float = 1.):
        super().__init__()

        self.initial_minimum_learning_rate = minimum_learning_rate
        self.initial_maximum_learning_rate = maximum_learning_rate
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.learning_rate_decay = learning_rate_decay
        self.restart_period_decay = restart_period_decay

        self.initial_samples_per_restart = self.epoch_size * epochs_per_restart
        self.samples_since_restart = 0
        self.restarts = 0.

        self.maximum_learning_rate = self.initial_maximum_learning_rate
        self.samples_per_restart = self.initial_samples_per_restart

    def restart(self):
        self.restarts += 1.
        self.samples_since_restart -= self.samples_per_restart
        # self.samples_since_restart = 0  # alternative, but the above line is more accurate over a large duration
        self.maximum_learning_rate = (self.learning_rate_decay ** self.restarts) * self.initial_maximum_learning_rate
        self.samples_per_restart = int((self.restart_period_decay ** self.restarts) * self.initial_samples_per_restart)

    def cosine_annealing(self):
        return self.initial_minimum_learning_rate + ((0.5 * (self.maximum_learning_rate - self.initial_minimum_learning_rate)) * (1 + np.cos((self.samples_since_restart / self.samples_per_restart) * np.pi)))

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_maximum_learning_rate)

    def on_batch_end(self, batch, logs=None):
        self.samples_since_restart += self.batch_size
        if self.samples_since_restart > self.samples_per_restart:
            self.restart()

        tf.keras.backend.set_value(self.model.optimizer.lr, self.cosine_annealing())

    def init_after_epochs(self, num_epochs):
        for epoch in range(num_epochs):
            for batch in range(int(self.epoch_size/self.batch_size)):
                self.on_batch_end(None)
            if self.epoch_size % self.batch_size != 0:
                self.on_batch_end(None)

