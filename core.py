from abc import ABC
import pandas as pd
import tensorflow
import tensorflow as tf
import ast
import numpy as np
import os
import json

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

# Configure distributed training across GPUs, if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


class ClassifierCore(ABC):
    def __init__(self, config):
        self.config = config

    def read_dataframe(self, path: str, confidence: float = 0.5, min_bbox_area: int = 3731,
                       min_class_img_count: int = 0, pixel_dilation: int = 5, train_mode: bool = True):
        """
        Reads and processes string path to CSV containing image paths and labels
        :param path: str, path to CSV file
        :param confidence: float, confidence level of object in bounding box of image
        :param min_bbox_area: int, min pixel area of bbox image, otherwise observation excluded
        :param min_class_img_count: int, minimum number of images per make-model, else this category is excluded
        :param pixel_dilation: int, number of extra pixels to add to YOLOv5 bounding box coordinates, producing less cropped image
        :param train_mode: bool, whether or not dataset should be initialized in train mode (i.e. uses existing labels in CSV),
            or test mode (i.e. looks for overlapping labels in CSV vs. label_mapping.json to correctly order softmax activations).
        :return: pd.DataFrame
        """
        df = pd.read_csv(path, usecols=['Make', 'Model', 'Source Path', 'Bboxes', 'Dims'])
        df['Bboxes'] = df['Bboxes'].apply(lambda x: list(ast.literal_eval(x)))
        df['Dims'] = df['Dims'].apply(lambda x: list(ast.literal_eval(x)))
        df = df.loc[df.Bboxes.str.len() != 0].reset_index(drop=True)  # restrict to rows with bounding boxes

        # Restrict to images with bounding boxes meeting minimum confidence level
        df['conf'] = df['Bboxes'].apply(lambda x: x[4])
        df = df.loc[df['conf'] >= confidence].reset_index(drop=True)

        # Bbox image size in pixel area]
        area = df['Bboxes'].apply(lambda x: (x[3] - x[1]) * (x[2] - x[0])).astype(int)  # Format: xyxy
        df = df.loc[area >= min_bbox_area].reset_index(drop=True)

        # Add extra area to bounding boxes
        df['Bboxes'] = df['Bboxes'].apply(lambda x: [max(x[0] - pixel_dilation, 0), max(x[1] - pixel_dilation, 0),
                                                         x[2] + pixel_dilation, x[3] + pixel_dilation])

        def enforce_boundaries(x):
            """
            Enforces that added pixel dilation area doesn't exceed total image size
            :param x: pd.Series
            :return: pd.Series
            """
            return [x[3][0], x[3][1], min(x[4][1], x[3][2]), min(x[4][0], x[3][3])]

        df['Bboxes'] = df.apply(enforce_boundaries, axis=1)

        # Modify bounding boxes, originally xyxy
        df['Bboxes'] = df['Bboxes'].apply(lambda x: [x[1], x[0], x[3] - x[1], x[2] - x[
            0]])  # rearranging for Tensorflow's preferred [y, x, y2-y1, x2-x1]

        # Concatenate Make and Model as new variable
        df['Make-Model'] = df['Make'] + ' ' + df['Model']

        # Modify `Source Path` to make absolute path to each image file
        df['Source Path'] = df['Source Path'].apply(lambda x: self.config['data'] + '/' + x)

        # Rearrange
        df = df[['Make', 'Make-Model', 'Source Path', 'Bboxes', 'conf']]

        if train_mode:

            # Remove make-model combinations below minimum image count
            counts = df.groupby('Make-Model')['Source Path'].count().reset_index()
            excluded = counts[counts['Source Path'] < min_class_img_count]['Make-Model'].tolist()
            df = df.loc[~df['Make-Model'].isin(excluded)]

            # Convert Make-Model to int, generate label mapping
            df['Make-Model'] = df['Make-Model'].astype('category')

            temp = dict(zip(df['Make-Model'].drop_duplicates().cat.codes, df['Make-Model'].cat.categories))

            # Attach category of vehicle to mapping
            label_mapping = {}
            for key, val in temp.items():
                label_mapping[key] = val

            df['Make-Model'] = df['Make-Model'].astype('category').cat.codes  # Converts to int

            # Convert Make-Model to onehot encodings
            dummies = pd.get_dummies(df['Make-Model'], prefix='')
            dummies.columns = [i[1:] for i in dummies.columns]

            # Concat together
            df = pd.concat([df[['Source Path', 'Bboxes', 'conf']], dummies], axis=1)

        else:  # predict mode

            # Verify overlap of labels between training and test data
            # NOTE - In predict mode, original TRAIN label mapping is returned
            with open(os.path.join(self.config['weights'], '../logs', 'label_mapping.json')) as f:
                label_mapping = json.load(f)

            labels = pd.DataFrame().from_dict(label_mapping, orient='index').rename(columns={0: 'Make-Model'})
            labels = labels.merge(df[['Make-Model']].drop_duplicates(), how='outer', indicator=True)
            assert (len(labels[labels._merge == 'right_only']) == 0), 'There are labels in test dataset not found in training data! Please drop these ahead of time'
            labels = labels.reset_index()

            # Convert Make-Model to int, generate label mapping
            df['Make-Model'] = df['Make-Model'].astype('category').cat.codes  # Converts to int

            # Convert Make-Model to onehot encodings
            dummies = pd.get_dummies(df['Make-Model'], prefix='')
            overlapping_labels = labels[labels._merge == 'both']['index'].tolist()
            dummies.columns = [i for i in overlapping_labels]  # relabel to get order right

            non_overlapping_labels = labels[labels._merge == 'left_only']['index'].tolist()
            for i in non_overlapping_labels:  # add in non-overlapping columns, all zero
                dummies[i] = 0

            dummies = dummies[sorted([i for i in dummies.columns])]  # specify correct sort order
            assert (dummies.shape[1] == len(label_mapping))

            # Concat together
            df = pd.concat([df[['Source Path', 'Bboxes', 'conf']], dummies], axis=1)

            # Covert label mapping values to int, rather than string
            label_mapping = {int(k):v for k,v in label_mapping.items()}

        # Shuffle data
        df = df.sample(frac=self.config['sample'], random_state=self.config['seed']).reset_index(drop=True)

        # Create stratified random sample
        if self.config['sample'] < 1.0:
            reverse_onehot = df.iloc[:, 2:].idxmax(axis=1).astype(int).reset_index()  # recover argmax
            indices = reverse_onehot.groupby(by=0, group_keys=False).apply(lambda x: x.sample(max(int(np.ceil(len(x) * self.config['sample'])), 1))).index
            df = df[df.index.isin(indices)]
            df = df.reset_index(drop=True)

        return df, label_mapping

    def load(self, image_file: tf.Tensor, channels: int = 0):
        """
        :param image_file: tf.Tensor containing string path to image
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

    def normalize(self, image: tf.Tensor):
        """
        Normalizes pixel values to 0-1 rage.
        :param image: 3d tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return (image / 255.0)

    def random_flip(self, image: tf.Tensor):
        """
        Randomly flip an image horizontally (left to right) deterministically.
        :param image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.random_flip_left_right(image)

    def random_brightness(self, image: tensorflow.Tensor):
        """
        Adjust the brightness of images by a random factor deterministically.
        :param image:  3d tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.random_brightness(image, max_delta=0.2)

    def random_contrast(self, image: tensorflow.Tensor):
        """
        Adjust the contrast of images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.random_contrast(image, lower=0.2, upper=0.5)

    def random_hue(self, image: tensorflow.Tensor):
        """
        Adjust the hue of RGB images by a random factor deterministically.
        :param image: 3d tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.random_hue(image, max_delta=0.2)

    def random_saturation(self, image: tf.Tensor):
        """
        Adjust the saturation of RGB images by a random factor deterministically.
        :param image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels]
        :return: A tensor of the same type and shape as image.
        """
        return tf.image.random_saturation(image, lower=0.2, upper=0.5)

    def bbox_crop(self, image: tf.Tensor, offset_height: int, offset_width: int, target_height: int, target_width: int):
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

    def grayscale_to_rgb(self, image: tf.Tensor):
        """
        Converts one or more images from Grayscale to RGB. Outputs a tensor of the same DType and rank as images.
        The size of the last dimension of the output is 3, containing the RGB value of the pixels. The input images'
        last dimension must be size 1.
        :param image: The Grayscale tensor to convert. The last dimension must be size 1.
        :return: 3-d tensor
        """
        return tf.image.grayscale_to_rgb(image)