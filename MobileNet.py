from core import ClassifierCore
import os
import argparse
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
import json
import pandas as pd
import matplotlib.pyplot as plt
from time import time

"""
    Credit: 
        https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
"""


class MobileNetClassifier(ClassifierCore):
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

    def image_pipeline(self, predict=False):
        print("\nReading in and processing images.\n", flush=True)

        if predict:  # TODO
            train = None
            validation = None
            test = None
        else:

            # Partition df into test, validation, and train splits, where train is x% RGB and 1-x% greyscale
            test = self.df.sample(frac=self.config['test_size'], random_state=self.config['seed'])
            df = self.df[~self.df.index.isin(test.index)]
            validation = df.sample(frac=self.config['validation_size'], random_state=self.config['seed'])
            train = df[~df.index.isin(validation.index)]

            # Convert to tensorflow dataset
            test = tf.data.Dataset.from_tensor_slices(
                (test['Source Path'], tf.cast(list(test['Bboxes']), tf.int32),
                 (test.iloc[:, 2:])))

            validation = tf.data.Dataset.from_tensor_slices(
                (validation['Source Path'], tf.cast(list(validation['Bboxes']), tf.int32),
                 (validation.iloc[:, 2:])))

            train = tf.data.Dataset.from_tensor_slices(
                (train['Source Path'], tf.cast(list(train['Bboxes']), tf.int32),
                 (train.iloc[:, 2:])))

            # Mapping function to read and adjust images
            # Note - large datasets should not be cached since cannot all fit in memory at once
            test = test.map(self.process_image, num_parallel_calls=tf.data.AUTOTUNE)
            validation = validation.map(self.process_image, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.map(self.process_image_train, num_parallel_calls=tf.data.AUTOTUNE)

            # Prefetch and batch
            train = train.batch(self.config['global_batch_size'])
            validation = validation.batch(self.config['global_batch_size'])
            test = test.batch(self.config['global_batch_size'])
            train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
            validation = validation.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train, validation, test

    def build_model(self):
        """
        Returns tf.keras.model that is not yet compiled
        :return:
        """

        # Instantiate MobileNetv2 layer
        try:
            mobilenet_layer = mobilenet_v2.MobileNetV2(input_shape=(self.config['img_size'] + (3,)),
                                             include_top=False,
                                             weights=f"./pretrained_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{self.config['mobilenetv2_alpha']}_{self.config['img_size'][0]}_no_top.h5")
        except ValueError:
            mobilenet_layer = mobilenet_v2.MobileNetV2(input_shape=(self.config['img_size'] + (3,)),
                                             include_top=False,
                                             weights=f"./scripts/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{self.config['mobilenetv2_alpha']}_{self.config['img_size'][0]}_no_top.h5")

        mobilenet_layer.trainable = False

        # Build model that includes MobileNetv2 layer
        inputs = tf.keras.Input(shape=self.config['img_size'] + (3,))
        x = mobilenet_v2.preprocess_input(inputs)  # handles image normalization
        x = mobilenet_layer(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        x = tf.keras.layers.Dense(self.config['units'], activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        output = tf.keras.layers.Dense(self.df.iloc[:, 2:].shape[1], activation='softmax')(x)
        model = tf.keras.Model(inputs, output)

        return model


    def train_model(self, train: tf.Tensor, validation: tf.Tensor, test: tf.Tensor, checkpoint_directory: str):

        performance_metrics = {}
        log_dir = os.path.join(checkpoint_directory, '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        with self.strategy.scope():

            def train_step(images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = compute_loss(labels, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                training_loss.update_state(loss)
                train_accuracy.update_state(labels, predictions)
                return loss

            def validation_step(images, labels):
                predictions = model(images, training=False)
                v_loss = loss_object(labels, predictions)

                validation_loss.update_state(v_loss)
                validation_accuracy.update_state(labels, predictions)

            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = self.strategy.run(train_step, args=(dataset_inputs))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            @tf.function
            def distributed_validation_step(dataset_inputs):
                return self.strategy.run(validation_step, args=(dataset_inputs))

            loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config['global_batch_size'])

            validation_loss = tf.keras.metrics.Mean(name='val_loss')
            training_loss = tf.keras.metrics.Mean(name='loss')
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
            validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

            model = self.build_model()

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'],
                                 beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])

            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        if self.config['save_weights'] == 'true':
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)

        start = time()
        for epoch in range(self.config['epochs']):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            for x in train:
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            # VALIDATION LOOP
            for x in validation:
                distributed_validation_step(x)

            if (epoch+1 % 5 == 0) and (epoch+1 != self.config['epochs']):  # save every fifth epoch's weights, so long as this not last epoch
                if self.config['save_weights'] == 'true':
                    checkpoint_manager.save()
            if epoch+1 == self.config['epochs']:  # save on last epoch
                if self.config['save_weights'] == 'true':
                    checkpoint_manager.save()

            template = ("Epoch {}, Cumulative Runtime (min) {:.2f} Loss: {:.4f}, Accuracy: {:.4f}, Validation Loss: {:.4f}, "
                        "Validation Accuracy: {:.4f}")
            print(template.format(epoch + 1, (time()-start)/60, train_loss,
                                  train_accuracy.result(), validation_loss.result(),
                                  validation_accuracy.result()))

            performance_metrics[epoch+1] = [train_loss.numpy(), train_accuracy.result().numpy(),
                                           validation_loss.result().numpy(),
                                           validation_accuracy.result().numpy()]

            validation_loss.reset_states()
            train_accuracy.reset_states()
            validation_accuracy.reset_states()

        df = pd.DataFrame(columns=['Loss', 'Accuracy', 'Val Loss', 'Val Accuracy']).from_dict(performance_metrics, orient='index', columns=['Loss', 'Accuracy', 'Val Loss', 'Val Accuracy'])
        df.to_csv(os.path.join(log_dir, 'metrics.csv'), index=True)


        # Evaluate on unseen data
        test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

        @tf.function
        def eval_step(images, labels):
            predictions = model(images, training=False)
            test_accuracy(labels, predictions)

        for images, labels in test:
            eval_step(images, labels)

        print("Categorical Accuracy on unseen data: {:.4f}".format(test_accuracy.result().numpy()))


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
        plt.ylabel('Categorical Crossentropy Loss')
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
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per replica. This number will be multiplied with the number of devices to yield global batch size')
    parser.add_argument('--logging', type=str, choices=['true', 'false'], default='true', help='turn off/on script logging, e.g. for CLI debugging')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
    parser.add_argument('--min-bbox-area', type=int, default=10000, help='minimum pixel area of bounding box, otherwise image excluded')
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
    parser.add_argument('--mobilenetv2-alpha', type=str, default='1.0', choices=['1.0', '0.75', '0.5', '0.35'], help='width multiplier in the MobileNetV2, options are 1.0, 0.75, 0.5, or 0.35')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Adam optimizer learning rate')
    parser.add_argument('--beta-1', type=float, default=0.9, help='exponential decay for first moment of Adam optimizer')
    parser.add_argument('--beta-2', type=float, default=0.999, help='exponential decay for second moment of Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout share in model')
    parser.add_argument('--units', type=int, default=1024, help='number of fully-connected units in second to last dense layer')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    args = parser.parse_args()
    assert (args.share_grayscale >= 0.0 and args.share_grayscale <= 1.0), "share-greyscale is bounded between 0-1!"
    assert (args.confidence >= 0.0 and args.confidence <= 1.0), "confidence is bounded between 0-1!"
    assert (args.validation_size >= 0.0 and args.validation_size <= 1.0), "validation size is a proportion and bounded between 0-1!"
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
    if opt.logging == 'true':
        sys.stdout = open(os.path.join(log_dir, "Log.txt"), "w")
        sys.stderr = sys.stdout

    mnc = MobileNetClassifier(vars(opt))

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(mnc.config, f)

    if opt.predict:
        pass

    else:
        train, validation, test = mnc.image_pipeline(predict=False)

        mnc.train_model(train, validation, test, checkpoint_directory=os.path.join(full_path, 'training_checkpoints'))

        # Generate performance metrics by epoch
        df = pd.read_csv(os.path.join(log_dir, 'metrics.csv')).set_index('Unnamed: 0')
        output_path = os.path.join(log_dir, '..', 'figs')
        os.makedirs(output_path, exist_ok=True)  # Creates output directory if not existing

        make_fig(train=df['Loss'], val=df['Val Loss'], output_path=output_path, loss=True)
        make_fig(train=df['Accuracy'], val=df['Val Accuracy'], output_path=output_path, loss=False)



if __name__ == '__main__':

    opt = parse_opt()
    main(opt)