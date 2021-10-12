from abc import ABC, abstractmethod
import tensorflow as tf
import os


tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

# Configure distributed training across GPUs, if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
if tf.config.list_physical_devices('GPU'):

    # Limit memory usage
    for dev in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(dev, True)

    strategy = tf.distribute.MirroredStrategy()  # uses all GPUs available in container

else:  # Use the Default Strategy
    strategy = tf.distribute.OneDeviceStrategy('/CPU:0')  # use for debugging


class ClassifierCore(ABC):
    def __init__(self, config):
        self.config = config
        self.config['global_batch_size'] = self.config['batch_size'] * strategy.num_replicas_in_sync
        self.strategy = strategy

    def load(self, image_file, channels=1):
        """
        :param image_file: str, absolute path to image file
        :param resize: bool, whether to resize image on read in to ensure consistently-sized images in tensor
        :return: A 3d tensor of shape [height, width, channels]
        """
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        try:
            image = tf.image.decode_png(image, channels=channels)
        except:
            image = tf.image.decode_jpeg(image, channels=channels)

        # Cast to float32 tensors
        image = tf.cast(image, tf.float32)

        return image

    def resize(self, image, height, width):
        """
        :param image: 3d tensor of shape [height, width, channels]
        :param height: int, pixel height
        :param width: int, pixel width
        :return: A tensor of desired resized dimensions
        """
        return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Normalizing the images to [0, 1]
    def normalize(self, image):
        """
        :param image: 3d tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return (image / 255.0)

    def random_flip(self, image, seed):
        """
        Randomly flip an image horizontally (left to right) deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: int, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_flip_left_right(image, seed)

    def random_brightness(self, image, seed):
        """
        Adjust the brightness of images by a random factor deterministically.
        :param image:  3d tensor of shape [height, width, channels]
        :param seed: int, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)

    def random_contrast(self, image, seed):
        """
        Adjust the contrast of images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: int, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_contrast(image, lower=0.2, upper=0.5, seed=seed)

    def random_hue(self, image, seed):
        """
        Adjust the hue of RGB images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: int, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed)

    def random_saturation(self, image, seed):
        """
        Adjust the saturation of RGB images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: int, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_saturation(image, lower=0.2, upper=0.5, seed=seed)


