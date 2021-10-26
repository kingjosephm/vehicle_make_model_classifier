from core import ClassifierCore
import os
import argparse
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import visualkeras
import numpy as np

"""
    Credit: 
        https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
"""


class MakeModelClassifier(ClassifierCore):
    def __init__(self, config):
        super().__init__(config)
        self.df, self.label_mapping = super().read_dataframe(self.config['img_df'])  # TODO - how handle predict mode?

    def process_image_train(self, image_file, bboxes: tf.Tensor, labels: tuple):
        """
        Loads, augments, and processes a single grayscale PNG/JPG image into tensor. Note - for tf.Keras.MobileNet
        input tensors should not be normalized.
        :param image_file: str, absolute path to PNG/JPG image
        :param bboxes: tf.Tensor tf.int32, bounding box coordinates
        :param labels: tuple, tuple of 4 tf.Tensor containing labels
        :return: 3d tensor of shape [height, width, channels]
        """
        # Randomly read in as grayscale or RGB
        gray = tf.keras.backend.random_bernoulli(shape=(1,), p=self.config['share_grayscale'])
        if gray == 1.0:
            image = super().load(image_file, channels=1)
        else:
            image = super().load(image_file, channels=3)

        image = super().bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        image = super().random_flip(image)
        image = super().random_brightness(image)
        image = super().random_contrast(image)
        if gray == 1.0:
            image = super().grayscale_to_rgb(image)
        else:
            image = super().random_hue(image)
            image = super().random_saturation(image)
        return image, labels

    def process_image(self, image_file: tf.Tensor, bboxes: tf.Tensor, labels: tuple):
        """
        Loads and processes a single color validation PNG/JPG image into tensor. Note - for tf.Keras.MobileNet
        input tensors should not be normalized.
        :param image_file: str, absolute path to PNG/JPG image
        :return: 3d tensor of shape [height, width, channels]
        """
        image = super().load(image_file)
        image = super().bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
        image = super().resize(image, height=self.config['img_size'][0], width=self.config['img_size'][1])
        return image, labels

    def create_balanced_df(self, df: pd.DataFrame):
        """
        Samples elements at random from the datasets in df
        :param df: pd.DataFrame
        :return: tensorflow.python.data.experimental.ops.interleave_ops._DirectedInterleaveDataset
        """
        categories = df.columns.tolist()[2:]
        df_list = []
        for x in categories:
            temp = df.loc[df[x] == 1]
            tf_df = tf.data.Dataset.from_tensor_slices(
                (temp['Source Path'], tf.cast(list(temp['Bboxes']), tf.int32), (temp.iloc[:, 2:])))
            tf_df = tf_df.shuffle(buffer_size=len(df)).repeat()
            df_list.append(tf_df)

        balanced_train = tf.data.experimental.sample_from_datasets(df_list, weights=[(1 / len(df_list))] * len(df_list))

        return balanced_train

    def image_pipeline(self, predict=False):
        print("\nReading in and processing images.\n", flush=True)

        if predict:  # TODO
            train = None
            validation = None
            test = None
        else:

            # Partition df into test, validation, and train splits, where train is x% RGB and 1-x% greyscale
            if self.config['test_size'] != 0:
                test = self.df.sample(frac=self.config['test_size'], random_state=self.config['seed'])
            else:
                test = pd.DataFrame()
            df = self.df[~self.df.index.isin(test.index)]
            validation = df.sample(frac=self.config['validation_size'], random_state=self.config['seed'])
            train = df[~df.index.isin(validation.index)]

            # Convert to tensorflow dataframe
            if self.config['balance_batches'] == 'true':
                validation = self.create_balanced_df(validation)
                train = self.create_balanced_df(train)
            else:
                validation = tf.data.Dataset.from_tensor_slices(
                    (validation['Source Path'], tf.cast(list(validation['Bboxes']), tf.int32), (validation.iloc[:, 2:])))
                train = tf.data.Dataset.from_tensor_slices(
                    (train['Source Path'], tf.cast(list(train['Bboxes']), tf.int32), (train.iloc[:, 2:])))

            test = tf.data.Dataset.from_tensor_slices(
                (test['Source Path'], tf.cast(list(test['Bboxes']), tf.int32), (test.iloc[:, 2:])))

            # Mapping function to read and adjust images
            # Note - large datasets should not be cached since cannot all fit in memory at once
            test = test.map(self.process_image, num_parallel_calls=tf.data.AUTOTUNE)
            validation = validation.map(self.process_image, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.map(self.process_image_train, num_parallel_calls=tf.data.AUTOTUNE)

            # Prefetch and batch
            train = train.batch(self.config['batch_size']).prefetch(buffer_size=tf.data.AUTOTUNE)
            validation = validation.batch(self.config['batch_size']).prefetch(buffer_size=tf.data.AUTOTUNE)
            test = test.batch(self.config['batch_size'])

        return train, validation, test

    def build_model(self):
        """
        Returns tf.keras.model that is not yet compiled
        :return:
        """

        # Instantiate MobileNetv2 layer
        mobilenet_layer = mobilenet_v2.MobileNetV2(input_shape=(self.config['img_size'] + (3,)),
                                         include_top=False,
                                         alpha=float(self.config['mobilenetv2_alpha']))


        # Set whole model mobilenet model to trainable or not
        if self.config['train_base'] == 'true':
            mobilenet_layer.trainable = True  # Note - keep training=False in mobilenet_layer below, so that this layer runs in inference mode so batchnorm stats don't update
        else:
            mobilenet_layer.trainable = False

        # Set last few layers as trainable
        if self.config['train_blocks'] < 0:
            if self.config['train_blocks'] == -1:
                train_blocks = -11
            elif self.config['train_blocks'] == -2:
                train_blocks = -20
            elif self.config['train_blocks'] == -3:
                train_blocks = -29
            else:
                raise ValueError("Please try training entire network")

            for layer in mobilenet_layer.layers[train_blocks:]:
                layer.trainable = True

        # Build model that includes MobileNetv2 layer
        inputs = tf.keras.Input(shape=self.config['img_size'] + (3,))
        x = mobilenet_v2.preprocess_input(inputs)  # handles image normalization
        x = mobilenet_layer(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        if self.config['units'] > 0:
            x = tf.keras.layers.Dense(self.config['units'], activation='relu')(x)
            x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        output = tf.keras.layers.Dense(self.df.iloc[:, 2:].shape[1], activation='softmax')(x)
        model = tf.keras.Model(inputs, output)

        return model


    def train_model(self, train: tf.Tensor, validation: tf.Tensor, checkpoint_directory: str):

        # Compile model
        model = self.build_model()

        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'],
                                             beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])

        model.compile(loss=loss_object, optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])


        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          restore_best_weights=True,
                                                          patience=self.config['patience'])

        if self.config['save_weights'] == 'true':
            os.makedirs(checkpoint_directory)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_directory+'/training_checkpoints', save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss')

        # Must be defined if balancing batches
        if self.config['balance_batches'] == 'true':
            train_steps_per_epoch = int(np.ceil((len(self.df) * (1 - (self.config['test_size'] + self.config['validation_size']))) / self.config['batch_size']))
            val_steps_per_epoch = int(np.ceil((len(self.df) * (1-self.config['test_size']) * self.config['validation_size']) / self.config['batch_size']))
        else:
            train_steps_per_epoch = None
            val_steps_per_epoch = None

        # Train model
        start = time()
        if self.config['save_weights'] == 'true':
            hist = model.fit(train, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
                             steps_per_epoch=train_steps_per_epoch, callbacks=[early_stopping, checkpoint],
                             validation_data=validation, validation_steps=val_steps_per_epoch)
        else:
            hist = model.fit(train, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
                             steps_per_epoch=train_steps_per_epoch, callbacks=[early_stopping],
                             validation_data=validation, validation_steps=val_steps_per_epoch)
        print("\nTotal training time in minutes: {:.2f}\n".format((time()-start)/60))
        return hist, model



def make_fig(train: pd.Series, val: pd.Series, output_path: str, loss: bool =True):
    '''
    Creates two line graphs in same figure using Matplotlib. Outputs as PNG to disk.
    :param train: pd.Series, loss/accuracy metrics by epoch for training set
    :param val:  pd.Series, loss/accuracy metrics by epoch for validation set
    :param output_path: str, path to output PNG
    :return: None, writes figure to disk
    '''
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(train, label='Train', linewidth=2)
    plt.plot(val, label='Validation', linewidth=2)
    plt.xlabel('Epochs')
    if loss:
        plt.ylabel('Crossentropy Loss')
        title = 'Loss by Epoch'
    else:
        plt.ylabel('Categorical Accuracy')
        title = 'Accuracy by Epoch'
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{''.join(title.split()[0])}.png"), dpi=200)
    plt.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    # Apply to train or predict modes
    parser.add_argument('--img-df', type=str, help='path to dataframe containing relative image paths and labels', required=True)  # TODO - accept dir path for predict?
    parser.add_argument('--data', type=str, default='./data/scraped_images', help='path to root directory where scraped vehicle images stored')
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=tuple, default=(224, 224), help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--logging', type=str, choices=['true', 'false'], default='true', help='turn off/on script logging, e.g. for CLI debugging')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
    parser.add_argument('--min-bbox-area', type=int, default=4000, help='minimum pixel area of bounding box, otherwise image excluded')
    parser.add_argument('--sample', type=float, default=1.0, help='share of image-df rows to sample, default 1.0. Used for debugging')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--validation-size', type=float, default=0.2, help='validation set size as share of number of training images')
    parser.add_argument('--test-size', type=float, default=0.05, help='holdout test set size as share of number of training images')
    parser.add_argument('--save-weights', type=str, choices=['true', 'false'], default='true', help='save model checkpoints and weights')
    parser.add_argument('--share-grayscale', type=float, default=0.5, help='share of training images to read in as greyscale')
    parser.add_argument('--confidence', type=float, default=0.70, help='object confidence level for YOLOv5 bounding box')
    parser.add_argument('--mobilenetv2-alpha', type=str, default='1.0', choices=['1.4', '1.3', '1.0', '0.75', '0.5', '0.35'], help='width multiplier in the MobileNetV2, options are 1.4, 1.0, 0.75, 0.5, or 0.35')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Adam optimizer learning rate')
    parser.add_argument('--beta-1', type=float, default=0.9, help='exponential decay for first moment of Adam optimizer')
    parser.add_argument('--beta-2', type=float, default=0.999, help='exponential decay for second moment of Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout share in model')
    parser.add_argument('--units', type=int, default=0, help='number of hidden units in last dense layer before output layer. Only applies if >0')
    parser.add_argument('--patience', type=int, default=5, help='patience parameter for model early stopping')
    parser.add_argument('--balance-batches', type=str, default='false', choices=['true', 'false'], help='whether or not to balance classes per mini batch')
    parser.add_argument('--train-base', type=str, default='false', choices=['true', 'false'], help="whether or not to unfreeze entire pretrained base model")
    parser.add_argument('--train-blocks', type=int, default=0, help="number of residual blocks at end of MobileNet to train, e.g. -1, -2 for last one or two blocks, respectively")
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    args = parser.parse_args()
    assert (args.share_grayscale >= 0.0 and args.share_grayscale <= 1.0), "share-greyscale is bounded between 0-1!"
    assert (args.confidence >= 0.0 and args.confidence <= 1.0), "confidence is bounded between 0-1!"
    assert (args.validation_size >= 0.0 and args.validation_size <= 1.0), "validation size is a proportion and bounded between 0-1!"
    assert (args.img_size == (224, 224)), "image size is only currently supported for 224 by 224 pixels"
    if args.train_base == 'true':
        if args.learning_rate > 1e-4:
            print("Warning - with base model set to trainable small learning rate (e.g. 1e-4) should be used")
    assert (args.train_blocks <= 0), "train-blocks are negative integers or 0 for None"
    if (args.train_base == 'true') and (args.train_blocks != 0):
        raise ValueError('Incompatible arguments! You can either train entire MobileNet model with `train-base` == `true` or e.g. `train-base` == -1 but not both!')
    return args

def main(opt):
    """
    Runs script using CLI arguments provided in opt.
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
    if opt.logging == 'true':
        sys.stdout = open(os.path.join(log_dir, "Log.txt"), "w")
        sys.stderr = sys.stdout

    mnc = MakeModelClassifier(vars(opt))

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(mnc.config, f)

    if opt.predict:
        pass

    else:
        train, validation, test = mnc.image_pipeline(predict=False)

        hist, model = mnc.train_model(train, validation, checkpoint_directory=os.path.join(full_path, 'training_checkpoints'))

        # Output figure of model structure to disk
        visualkeras.layered_view(model, legend=True, to_file=os.path.join(log_dir, 'model_structure.png'))
        visualkeras.layered_view(model, legend=True, scale_xy=1, scale_z=1, max_z=1000, to_file=os.path.join(log_dir, 'model_structure_scaled.png'))

        # Evaluate using test set
        if opt.test_size != 0:
            results = model.evaluate(test, batch_size=opt.batch_size)
            print("Model results in unseen data: Loss {:.3f}, Accuracy {:.3f}".format(results[0], results[1]))

        predictions = model.predict(test)

        categories = []
        label_mapping_short = {}
        for key, val in mnc.label_mapping.items():
            categories.append(val[0])
            label_mapping_short[key] = val[0]
        pred_df = pd.DataFrame(predictions, columns=categories)

        # Recover labels
        images, labels = tuple(zip(*test))

        label_df = pd.DataFrame()
        for x in range(len(labels)):
            label_df = pd.concat([label_df, pd.DataFrame(labels[x].numpy())], axis=0)
        label_df = label_df.reset_index(drop=True)
        label_series = label_df.idxmax(axis=1)
        label_series.replace(to_replace=label_mapping_short, inplace=True)

        pred_df = pd.concat([pred_df, label_series], axis=1).rename(columns={0: 'true_label'})

        pred_df.to_csv(os.path.join(log_dir, 'predictions.csv'), index=False)


        # Get model performance by epoch
        df = pd.DataFrame().from_dict(hist.history, orient='columns').reset_index()
        df['index'] = df['index'] + 1
        df.rename(columns={'index': 'epoch', 'categorical_accuracy': 'accuracy', 'val_categorical_accuracy': 'val_accuracy'}, inplace=True)
        df.to_csv(os.path.join(log_dir, 'metrics.csv'), index=True)

        # Generate performance metric figures
        df = df.set_index('epoch')
        output_path = os.path.join(log_dir, '..', 'figs')
        os.makedirs(output_path, exist_ok=True)  # Creates output directory if not existing

        make_fig(train=df['loss'], val=df['val_loss'], output_path=output_path, loss=True)
        make_fig(train=df['accuracy'], val=df['val_accuracy'], output_path=output_path, loss=False)


if __name__ == '__main__':

    opt = parse_opt()
    main(opt)