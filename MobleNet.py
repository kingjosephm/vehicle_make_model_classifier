from core import ClassifierCore
import os
import argparse
import sys
from datetime import datetime
import random
import math
import tensorflow as tf

class MobleNetClassifier(ClassifierCore):
    def __init__(self, config):
        super().__init__(config)

    def process_image_train(self, image_file):
        """
        Loads, augments, and processes a single PNG/JPG image into tensor
        :param image_file: str, absolute path to PNG/JPG image
        :return: 3d tensor of shape [height, width, channels]
        """
        image = super().load(image_file, channels=self.config['channels'])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        image = super().random_flip(image, seed=self.config['seed'])
        image = super().random_brightness(image, seed=self.config['seed'])
        image = super().random_contrast(image, seed=self.config['seed'])
        if self.config['channels'] == '3':  # only applies to color images
            image = super().random_hue(image, seed=self.config['seed'])
            image = super().random_saturation(image, seed=self.config['seed'])
        image = super().normalize(image)
        return image


    def process_image_val(self, image_file):
        """
        Loads and processes a single PNG/JPG image into tensor.
        :param image_file: str, absolute path to PNG/JPG image
        :return: 3d tensor of shape [height, width, channels]
        """
        image = super().load(image_file, channels=self.config['channels'])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        return image

    def image_pipeline(self, predict=False):
        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents = [os.path.join(self.config['data'], i) for i in os.listdir(self.config['data']) if 'png' in i or 'jpg' in i]
        assert (contents), "No JPEG or PNG images found in data directory!"

        if predict:
            pass
        else:

            validation = random.sample(contents, math.ceil(len(contents) * self.config['validation_size']))
            validation = tf.data.Dataset.from_tensor_slices(validation)
            validation.map(self.process_image_val, num_parallel_calls=tf.data.AUTOTUNE)


def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--data', type=str, help='path to input images', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=tuple, default=(256, 256), help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per replica')
    parser.add_argument('--no-log', action='store_true', help='turn off script logging, e.g. for CLI debugging')
    parser.add_argument('--channels', type=str, default='1', choices=['1', '3'], help='number of color channels to read image using')
    parser.add_argument('--crop-image', type=bool, default=True, help='whether or not to crop input image using YOLOv5 prior to classification')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
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
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    return parser.parse_args()

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