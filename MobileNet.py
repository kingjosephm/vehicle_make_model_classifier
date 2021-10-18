from core import ClassifierCore
import os
import argparse
import sys
from datetime import datetime
import tensorflow as tf

class MobleNetClassifier(ClassifierCore):
    def __init__(self, config):
        super().__init__(config)
        self.df, \
        self.make, \
        self.make_model, \
        self.category = super().read_dataframe(self.config['img_dataframe_path'])  # TODO - how handle predict mode?

    def process_image_train_grey(self, image_file, bboxes):
        """
        Loads, augments, and processes a single PNG/JPG image into tensor
        :param image_file: str, absolute path to PNG/JPG image
        :return: 3d tensor of shape [height, width, channels]
        """
        image = super().load(image_file, channels=1)
        image = super().bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        image = super().random_flip(image, seed=(1, self.config['seed']))
        image = super().random_brightness(image, seed=(1, self.config['seed']))
        image = super().random_contrast(image, seed=(1, self.config['seed']))
        image = super().grayscale_to_rgb(image)
        image = super().normalize(image)
        return image

    def process_image_train_color(self, image_file, bboxes):
        image = super().load(image_file, channels=3)
        image = super().bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        image = super().random_flip(image, seed=(1, self.config['seed']))
        image = super().random_brightness(image, seed=(1, self.config['seed']))
        image = super().random_contrast(image, seed=(1, self.config['seed']))
        image = super().random_hue(image, seed=(1, self.config['seed']))
        image = super().random_saturation(image, seed=(1, self.config['seed']))
        image = super().normalize(image)
        return image


    def process_image_val(self, image_file, bboxes):
        """
        Loads and processes a single PNG/JPG image into tensor.
        :param image_file: str, absolute path to PNG/JPG image
        :return: 3d tensor of shape [height, width, channels]
        """
        image = super().load(image_file)
        image = super().bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        image = super().normalize(image)
        return image

    def image_pipeline(self, predict=False):
        print("\nReading in and processing images.\n", flush=True)

        if predict:  # TODO
            train = None
            validation = None
        else:

            # Partition df into validation and train splits, where train is x% RGB and 1-x% greyscale
            validation = self.df.sample(frac=self.config['validation_size'], random_state=self.config['seed'])
            train = self.df[~self.df.index.isin(validation.index)]
            rgb = train.sample(frac=(1-self.config['share_greyscale']), random_state=self.config['seed'])
            gray = train[~train.index.isin(rgb.index)]

            # Convert to tensorflow dataset
            val = tf.data.Dataset.from_tensor_slices(
                (validation['Source Path'], tf.cast(list(validation['Bboxes']), tf.int32)))
            color = tf.data.Dataset.from_tensor_slices((rgb['Source Path'], tf.cast(list(rgb['Bboxes']), tf.int32)))
            grayscale = tf.data.Dataset.from_tensor_slices(
                (gray['Source Path'], tf.cast(list(gray['Bboxes']), tf.int32)))

            # Mapping function to read and adjust images
            val = val.map(self.process_image_val, num_parallel_calls=tf.data.AUTOTUNE).cache()
            color = color.map(self.process_image_train_color, num_parallel_calls=tf.data.AUTOTUNE).cache()
            grayscale = grayscale.map(self.process_image_train_grey, num_parallel_calls=tf.data.AUTOTUNE).cache()

            # Attach labels to respective tf dataframes
            validation_labels = tf.data.Dataset.from_tensor_slices((validation['Make-Model'],
                                                                    validation['Make'], validation['Category']))
            color_labels = tf.data.Dataset.from_tensor_slices((rgb['Make-Model'], rgb['Make'], rgb['Category']))
            gray_labels = tf.data.Dataset.from_tensor_slices((gray['Make-Model'], gray['Make'], gray['Category']))

            validation = tf.data.Dataset.zip((val, validation_labels))
            color = tf.data.Dataset.zip((color, color_labels))
            grayscale = tf.data.Dataset.zip((grayscale, gray_labels))

            # Concatenate color and grayscale together
            train = color.concatenate(grayscale)

            # Shuffle data
            train = train.shuffle(buffer_size=len(train)*10, seed=self.config['seed'], reshuffle_each_iteration=True)
            validation = validation.shuffle(buffer_size=len(validation)*10, seed=self.config['seed'], reshuffle_each_iteration=True )

            # Prefetch and batch
            train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
            validation = validation.prefetch(buffer_size=tf.data.AUTOTUNE)
            train = train.batch(self.config['global_batch_size'])
            validation = validation.batch(self.config['global_batch_size'])

        return train, validation



def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--img-dataframe-path', type=str, help='path to dataframe containing image paths and labels', required=True)  # TODO - accept dir path for predict?
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=tuple, default=(256, 256), help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per replica')
    parser.add_argument('--no-log', action='store_true', help='turn off script logging, e.g. for CLI debugging')
    parser.add_argument('--crop-image', type=bool, default=True, help='whether or not to crop input image using YOLOv5 prior to classification')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
    parser.add_argument('--min-bbox-area', type=int, default=10000, help='minimum pixel area of bounding box, otherwise image excluded')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train', required='--train' in sys.argv)
    parser.add_argument('--validation-size', type=float, default=0.2, help='validation set size as share of number of training images', required='--train' in sys.argv)  # TODO - add int type too
    group2 = parser.add_mutually_exclusive_group(required='--train' in sys.argv)
    group2.add_argument('--save-weights', action='store_true', help='save model checkpoints and weights')
    group2.add_argument('--no-save-weights', action='store_true', help='do not save model checkpoints or weights')
    parser.add_argument('--share-greyscale', type=float, default=0.5, help='share of training images to read in as greyscale')
    parser.add_argument('--confidence', type=float, default=0.70, help='object confidence level for YOLOv5 bounding box')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    args = parser.parse_args()
    assert (args.share_greyscale >= 0.0 and args.share_greyscale <= 1.0), "share-greyscale is bounded between 0-1!"
    assert (args.confidence >= 0.0 and args.confidence <= 1.0), "confidence is bounded between 0-1!"
    return args

def main(opt):
    """
    :param opt: argparse.Namespace
    :return: None
    """

    # Directing output
    os.makedirs(opt.output, exist_ok=True)
    full_path = opt.output + '/' + datetime.now().strftime("%Y-%m-%d-%Hh%M")
    os.makedirs(full_path, exist_ok=True)  # will overwrite folder if model run within same minute

    # Log results
    log_dir = os.path.join(full_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if not opt.no_log:
        sys.stdout = open(os.path.join(log_dir, "Log.txt"), "w")
        sys.stderr = sys.stdout

    mnc = MobleNetClassifier(vars(opt))


if __name__ == '__main__':

    opt = parse_opt()
    main(opt)