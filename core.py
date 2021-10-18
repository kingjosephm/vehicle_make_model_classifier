from abc import ABC, abstractmethod
import pandas as pd
import tensorflow
import tensorflow as tf
import ast

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

    def read_dataframe(path: str, confidence: float = 0.6, min_bbox_area: int = 10000):
        """
        Reads and processes string path to CSV containing image paths and labels
        :param path: str, path to CSV file
        :param confidence: float, confidence level of object in bounding box of image
        :param min_bbox_size: int, min pixel area of bbox image, otherwise observation excluded
        :return: pd.DataFrame
        """
        df = pd.read_csv(path, usecols=['Make', 'Model', 'Category', 'Source Path', 'Bboxes'])
        df['Bboxes'] = df['Bboxes'].apply(lambda x: list(ast.literal_eval(x)))
        df = df.loc[df.Bboxes.str.len() != 0].reset_index(drop=True)  # restrict to rows with bounding boxes

        # Restrict to images with bounding boxes meeting minimum confidence level
        conf = df['Bboxes'].apply(lambda x: x[4])
        df = df.loc[conf >= confidence].reset_index(drop=True)

        # Bbox image size in pixel area]
        area = df['Bboxes'].apply(lambda x: (x[3] - x[1]) * (x[2] - x[0])).astype(int)  # Format: xyxy
        df = df.loc[area >= min_bbox_area].reset_index(drop=True)

        # Modify bounding boxes, originally xyxy
        df['Bboxes'] = df['Bboxes'].apply(lambda x: [x[1], x[0], x[3] - x[1], x[2] - x[
            0]])  # rearranging for Tensorflow's preferred [y, x, y2-y1, x2-x1]

        # Concatenate Make and Model as new variable
        df['Make-Model'] = df['Make'] + ' ' + df['Model']

        # Rearrange
        df = df[['Make', 'Make-Model', 'Category', 'Source Path', 'Bboxes']]\
            .sample(frac=1.0, random_state=123).reset_index(drop=True)

        return df

    def load(self, image_file: str, channels: int = 0):
        """
        :param image_file: str, absolute path to image file
        :param channels, int, desired number of color channels for decoded image. By default it infers the number.
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

    def resize(self, image: tensorflow.Tensor, height: int, width: int):
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

    def random_flip(self, image: tensorflow.Tensor, seed: tuple = (1, 123)):
        """
        Randomly flip an image horizontally (left to right) deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: 2d tuple, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_flip_left_right(image, seed)

    def random_brightness(self, image: tensorflow.Tensor, seed: tuple = (1, 123)):
        """
        Adjust the brightness of images by a random factor deterministically.
        :param image:  3d tensor of shape [height, width, channels]
        :param seed: 2d tuple, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)

    def random_contrast(self, image: tensorflow.Tensor, seed: tuple = (1, 123)):
        """
        Adjust the contrast of images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: 2d tuple, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_contrast(image, lower=0.2, upper=0.5, seed=seed)

    def random_hue(self, image: tensorflow.Tensor, seed: tuple = (1, 123)):
        """
        Adjust the hue of RGB images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: 2d tuple, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed)

    def random_saturation(self, image: tensorflow.Tensor, seed: tuple = (1, 123)):
        """
        Adjust the saturation of RGB images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :param seed: 2d tuple, seed for random number generator
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.stateless_random_saturation(image, lower=0.2, upper=0.5, seed=seed)

    def bbox_crop(self, image, offset_height, offset_width, target_height, target_width):
        """
        Crops an image according to bounding box coordinates
        :param image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels]
        :param offset_height: Vertical coordinate of the top-left corner of the bounding box in image.
        :param offset_width: Horizontal coordinate of the top-left corner of the bounding box in image.
        :param target_height: Height of the bounding box.
        :param target_width: Width of the bounding box.
        :return: 4- or 3-D tensor
        """
        return tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

    def grayscale_to_rgb(self, image):
        """
        Converts one or more images from Grayscale to RGB. Outputs a tensor of the same DType and rank as images.
        The size of the last dimension of the output is 3, containing the RGB value of the pixels. The input images'
        last dimension must be size 1.
        :param image: The Grayscale tensor to convert. The last dimension must be size 1.
        :return: 3-d tensor
        """
        return tf.image.grayscale_to_rgb(image)