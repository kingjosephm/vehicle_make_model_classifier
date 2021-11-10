from core import ClassifierCore
import os
import argparse
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2, resnet_v2, xception, inception_v3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import visualkeras
import numpy as np
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

"""
    Credit: 
        https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
"""


class MakeModelClassifier(ClassifierCore):
    def __init__(self, config):
        super().__init__(config)
        self.df, self.label_mapping = super().read_dataframe(self.config['img_df'],
                                                             min_class_img_count=self.config['min_class_img_count'],
                                                             pixel_dilation=self.config['pixel_dilation'])  # TODO - how handle predict mode?

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
            # Partition df into test, validation, and train splits
            # Ensuring all three are balanced wrt classes by creating stratified random samples
            reverse_onehot = self.df.iloc[:, 2:].idxmax(axis=1).astype(int).reset_index()  # recover argmax

            test_indices = reverse_onehot.groupby(by=0, group_keys=False).apply(lambda x: x.sample(max(int(np.floor(len(x) * self.config['test_size'])), 2))).index

            remainder = reverse_onehot[~reverse_onehot.index.isin(test_indices)]

            validation_indices = remainder.groupby(by=0, group_keys=False).apply(
                lambda x: x.sample(max(int(np.floor(len(x) * self.config['validation_size'])), 2))).index

            train_indices = remainder[~remainder.index.isin(validation_indices)].index

            test = self.df[self.df.index.isin(test_indices)]
            validation = self.df[self.df.index.isin(validation_indices)]
            train = self.df[self.df.index.isin(train_indices)]

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

        if self.config['model'] == 'mobilenet':
            pretrained_layer = mobilenet_v2.MobileNetV2(input_shape=(self.config['img_size'] + (3,)),
                                             include_top=False,
                                             alpha=float(self.config['mobilenetv2_alpha']))

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

        elif self.config['model'] == 'resnet':
            pretrained_layer = resnet_v2.ResNet152V2(input_shape=(self.config['img_size'] + (3,)), include_top=False)

            # Set last few layers as trainable - TODO
            if self.config['train_blocks'] < 0:
                raise ValueError(f"This feature is currently not available for {self.config['model']}!")

        elif self.config['model'] == 'xception':
            pretrained_layer = xception.Xception(input_shape=(self.config['img_size'] + (3,)), include_top=False)

            # Set last few layers as trainable - TODO
            if self.config['train_blocks'] < 0:
                raise ValueError(f"This feature is currently not available for {self.config['model']}!")

        else:
            pretrained_layer = inception_v3.InceptionV3(input_shape=(self.config['img_size'] + (3,)), include_top=False)

            # Set last few layers as trainable - TODO
            if self.config['train_blocks'] < 0:
                raise ValueError(f"This feature is currently not available for {self.config['model']}!")


        if self.config['train_blocks'] < 0:
            for layer in pretrained_layer.layers[train_blocks:]:
                layer.trainable = True

        # Set whole model mobilenet model to trainable or not
        if self.config['train_base'] == 'true':
            pretrained_layer.trainable = True  # Note - keep training=False in mobilenet_layer below, so that this layer runs in inference mode so batchnorm stats don't update
        else:
            pretrained_layer.trainable = False

        # Build model that includes MobileNetv2 layer
        inputs = tf.keras.Input(shape=self.config['img_size'] + (3,))

        # handles image normalization & preprocessing
        if self.config['model'] == 'mobilenet':
            x = mobilenet_v2.preprocess_input(inputs)
        elif self.config['model'] == 'resnet':
            x = resnet_v2.preprocess_input(inputs)
        elif self.config['model'] == 'xception':
            x = xception.preprocess_input(inputs)
        else:  # inception
            x = inception_v3.preprocess_input(inputs)

        x = pretrained_layer(x, training=False)
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
    parser.add_argument('--confidence', type=float, default=0.50, help='object confidence level for YOLOv5 bounding box')
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'resnet', 'xception', 'inception'], help='pretrained model type, options are `mobilenet` (MobileNetV2), `resnet` (ResNet152V2), `xception` (Xception), or `inception` (InceptionV3)')
    parser.add_argument('--mobilenetv2-alpha', type=str, default='1.0', choices=['1.4', '1.3', '1.0', '0.75', '0.5', '0.35'], help='width multiplier in the MobileNetV2, options are 1.4, 1.3, 1.0, 0.75, 0.5, or 0.35')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Adam optimizer learning rate')
    parser.add_argument('--beta-1', type=float, default=0.9, help='exponential decay for first moment of Adam optimizer')
    parser.add_argument('--beta-2', type=float, default=0.999, help='exponential decay for second moment of Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout share in model')
    parser.add_argument('--units', type=int, default=0, help='number of hidden units in last dense layer before output layer. Only applies if >0')
    parser.add_argument('--patience', type=int, default=5, help='patience parameter for model early stopping')
    parser.add_argument('--balance-batches', type=str, default='false', choices=['true', 'false'], help='whether or not to balance classes per mini batch')
    parser.add_argument('--train-base', type=str, default='false', choices=['true', 'false'], help="whether or not to unfreeze entire pretrained base model")
    parser.add_argument('--train-blocks', type=int, default=0, help="number of residual blocks at end of MobileNet to train, e.g. -1, -2 for last one or two blocks, respectively")
    parser.add_argument('--min-class-img-count', type=int, default=0, help='minimum number of images per make-model, else discard this class')
    parser.add_argument('--pixel-dilation', type=int, default=5, help='number of pixels to add around YOLOv5 bounding box coordinates')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    args = parser.parse_args()
    assert (args.share_grayscale >= 0.0 and args.share_grayscale <= 1.0), "share-greyscale is bounded between 0-1!"
    assert (args.confidence > 0.0 and args.confidence <= 1.0), "confidence is bounded between 0-1!"
    assert (args.test_size > 0.0 and args.test_size <= 0.15), "test size is a proportion and bounded between 0-0.15!"
    assert (args.validation_size > 0.0 and args.validation_size <= 0.3), "validation size is a proportion and bounded between 0-0.3!"
    assert (args.img_size == (224, 224)), "image size is only currently supported for 224 by 224 pixels"
    assert (args.pixel_dilation >= 0), 'pixel dilation must be >= 0!'
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
    os.makedirs(full_path, exist_ok=False)  # must wait >1 min between running models to differentiate output directory

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
            print("\nModel results in unseen data: Loss {:.3f}, Accuracy {:.3f}".format(results[0], results[1]))

        # Dataframe of predictions
        predictions = model.predict(test)

        categories = []
        label_mapping_short = {}
        for key, val in mnc.label_mapping.items():
            categories.append(val[0])
            label_mapping_short[key] = val[0]
        pred_df = pd.DataFrame(predictions, columns=categories)

        images, labels = tuple(zip(*test))  # Recover labels

        label_df = pd.DataFrame()
        for x in range(len(labels)):
            label_df = pd.concat([label_df, pd.DataFrame(labels[x].numpy())], axis=0)
        label_df = label_df.reset_index(drop=True)
        label_series = label_df.idxmax(axis=1)
        label_series.replace(to_replace=label_mapping_short, inplace=True)

        pred_df = pd.concat([pred_df, label_series], axis=1).rename(columns={0: 'true_label'})

        pred_df.to_csv(os.path.join(log_dir, 'predictions.csv'), index=False)

        # Model performance by epoch
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

        # Rank accuracy figure
        true = pred_df['true_label'].copy()
        del pred_df['true_label']

        lst = []
        index = pred_df.columns.tolist()  # columns become indices below
        for i in range(len(pred_df)):
            top_five = np.argsort(pred_df.iloc[i].values)[-5:]
            names = list(reversed([index[i] for i in top_five]))
            lst.append(names)
        pred_classes = pd.DataFrame(lst, columns=['Argmax(0)', 'Argmax(1)', 'Argmax(2)', 'Argmax(3)', 'Argmax(4)'])

        classes = pd.concat([true, pred_classes], axis=1)

        one = np.where(classes['true_label'] == classes['Argmax(0)'], 1, 0)
        two = np.where((classes['true_label'] == classes['Argmax(0)']) |
                       (classes['true_label'] == classes['Argmax(1)']), 1, 0)
        three = np.where((classes['true_label'] == classes['Argmax(0)']) |
                         (classes['true_label'] == classes['Argmax(1)']) |
                         (classes['true_label'] == classes['Argmax(2)']), 1, 0)
        four = np.where((classes['true_label'] == classes['Argmax(0)']) |
                        (classes['true_label'] == classes['Argmax(1)']) |
                        (classes['true_label'] == classes['Argmax(2)']) |
                        (classes['true_label'] == classes['Argmax(3)']), 1, 0)
        five = np.where((classes['true_label'] == classes['Argmax(0)']) |
                        (classes['true_label'] == classes['Argmax(1)']) |
                        (classes['true_label'] == classes['Argmax(2)']) |
                        (classes['true_label'] == classes['Argmax(3)']) |
                        (classes['true_label'] == classes['Argmax(4)']), 1, 0)

        accuracy = pd.DataFrame()
        accuracy = pd.concat([accuracy, pd.DataFrame(one.mean(), columns=['Accuracy'], index=['Argmax(0)'])], axis=0)
        accuracy = pd.concat([accuracy, pd.DataFrame(two.mean(), columns=['Accuracy'], index=['Argmax(0:1)'])], axis=0)
        accuracy = pd.concat([accuracy, pd.DataFrame(three.mean(), columns=['Accuracy'], index=['Argmax(0:2)'])],
                             axis=0)
        accuracy = pd.concat([accuracy, pd.DataFrame(four.mean(), columns=['Accuracy'], index=['Argmax(0:3)'])], axis=0)
        accuracy = pd.concat([accuracy, pd.DataFrame(five.mean(), columns=['Accuracy'], index=['Argmax(0:4)'])], axis=0)
        accuracy = accuracy.reset_index()

        figure(figsize=(10, 8))
        g = sns.barplot(data=accuracy, x='index', y='Accuracy', palette='Set2')
        for index, row in accuracy.iterrows():  # print accuracy values atop each bar
            g.text(row.name, row.Accuracy, round(row.Accuracy, 4), color='black', ha='center')
        plt.xlabel(None)
        plt.ylabel('Categorical Accuracy')
        plt.title('Accuracy Among Top 5 Predicted Classes')
        plt.savefig(os.path.join(log_dir, 'Accuracy_Top5.png'))
        plt.close()

        # Multiclass confusion matrix
        labels = classes['true_label'].drop_duplicates().sort_values().tolist()
        conf_mat = pd.DataFrame(
            confusion_matrix(classes['true_label'], classes['Argmax(0)'], normalize='true', labels=labels),
            index=labels, columns=labels)
        conf_mat.to_csv(os.path.join(log_dir, 'confusion_matrix.csv'))

        # Output heatmap of confusion matrix
        figure(figsize=(25, 25))
        sns.set(font_scale=0.5)
        ax = sns.heatmap(conf_mat, cmap='Reds', linewidth=0.8, cbar_kws={"shrink": 0.8}, square=True)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.title('Confusion Matrix Heatmap', fontsize=30)
        plt.savefig(os.path.join(log_dir, 'heatmap.png'))
        plt.close()

        # One vs rest confusion matrix
        ml_conf_mat = multilabel_confusion_matrix(classes['true_label'], classes['Argmax(0)'], labels=labels,
                                                  samplewise=False)  # class-wise confusion matrix
        lst = []
        for i in range(len(ml_conf_mat)):
            temp = ml_conf_mat[i]
            tnr, fpr, fnr, tpr = temp.ravel()
            lst.append([tnr, fpr, fnr, tpr])
        ovr_conf_mat = pd.DataFrame(lst, columns=['TN', 'FP', 'FN', 'TP'], index=labels)
        ovr_conf_mat = ovr_conf_mat[['TP', 'FN', 'FP', 'TN']]
        ovr_conf_mat['Accuracy'] = (ovr_conf_mat['TP'] + ovr_conf_mat['TN']) / ovr_conf_mat.sum(axis=1).mean()
        ovr_conf_mat['Precision'] = ovr_conf_mat['TP'] / (ovr_conf_mat['TP'] + ovr_conf_mat['FP'])
        ovr_conf_mat['Recall_Sensitivity_TPR'] = ovr_conf_mat['TP'] / (ovr_conf_mat['TP'] + ovr_conf_mat['FN'])
        ovr_conf_mat['FNR'] = ovr_conf_mat['FN'] / (ovr_conf_mat['FN'] + ovr_conf_mat['TP'])
        ovr_conf_mat['FPR'] = ovr_conf_mat['FP'] / (ovr_conf_mat['FP'] + ovr_conf_mat['TN'])
        ovr_conf_mat['Specificity_TNR'] = ovr_conf_mat['TN'] / (ovr_conf_mat['TN'] + ovr_conf_mat['FP'])
        ovr_conf_mat['F1'] = 2 * ovr_conf_mat['TP'] / (
                    (2 * ovr_conf_mat['TP']) + ovr_conf_mat['FP'] + ovr_conf_mat['FN'])
        for col in ovr_conf_mat.columns[4:]:
            ovr_conf_mat[col] = round(ovr_conf_mat[col], 4)
        ovr_conf_mat.to_csv(os.path.join(log_dir, 'OVR Confusion Matrix.csv'))

        # Kernel density plot of sensitivity
        figure(figsize=(8, 8))
        sns.set(font_scale=1)
        sns.kdeplot(ovr_conf_mat['Recall_Sensitivity_TPR'])
        plt.xlabel('Sensitivity / Recall / TPR')
        plt.title("Kernel Density of Sensitivity")
        plt.savefig(os.path.join(log_dir, 'sensitivity_kdeplot.png'))
        plt.close()

        sens = ovr_conf_mat[['Recall_Sensitivity_TPR']].copy()
        sens = sens.sort_values(by=['Recall_Sensitivity_TPR'], ascending=False).reset_index()
        combined = pd.concat([sens.iloc[:50, :], sens.iloc[-50:, :]], axis=0).reset_index(drop=True)

        # Figure to output barplot of sensitivity among best and worst 50 classified make-models
        plt.close()
        figure(figsize=(20, 8))
        sns.set(font_scale=0.8)
        ax = sns.barplot(data=combined, y='Recall_Sensitivity_TPR', x='index', saturation=0.9)
        plt.xticks(rotation=70, ha='right')
        plt.yticks(fontsize=10)
        plt.xlabel(None)
        plt.ylabel('Sensitivity / Recall / TPR', fontsize=15)
        plt.tight_layout()
        plt.title('Best and Worst 50 Classified Make-Models', fontsize=20, pad=-18)
        plt.savefig(os.path.join(log_dir, 'sensitivity_bar.png'), dpi=200)
        plt.close()


if __name__ == '__main__':

    opt = parse_opt()
    main(opt)