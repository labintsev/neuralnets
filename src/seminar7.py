"""
Seminar 7. SMS spam Classification with Recurrent Nets.
https://www.kaggle.com/code/darthmanav/rnn-for-text-classification
https://www.tensorflow.org/guide/keras/working_with_rnns
"""
import argparse
import os
import shutil
from urllib.request import urlretrieve

import boto3
import dotenv
import pandas as pd
import tensorflow as tf
import pandas as pd
import tensorflow as tf
import argparse
import os
import shutil
from urllib.request import urlretrieve
import boto3
import dotenv
import logging
#####
MAX_WORDS = 4000
MAX_SEQ_LEN = 150
DATA_URL_TRAIN = 'https://storage.yandexcloud.net/fa-bucket/spam.csv'
DATA_URL_TEST = 'https://storage.yandexcloud.net/fa-bucket/spam_test.csv'
PATH_TO_TRAIN_DATA = 'data/raw/spam.csv'
PATH_TO_TEST_DATA = 'data/raw/spam_test.csv'
PATH_TO_MODEL = 'models/model_7'
BUCKET_NAME = 'neuralnets2023'
# todo fix your git user name
YOUR_GIT_USER = 'meribabayaan'


def download_data():
    """Pipeline: download and extract data"""
    if not os.path.exists(PATH_TO_TRAIN_DATA):
        print('Downloading data...')
        urlretrieve(DATA_URL_TRAIN, PATH_TO_TRAIN_DATA)
        urlretrieve(DATA_URL_TEST, PATH_TO_TEST_DATA)
    else:
        print('Data is already downloaded!')


def make_model():
    """
    Make recurrent model for binary classification.
    todo find good layers and hyperparameters
    :return:
    """
    inputs = tf.keras.layers.Input(name='inputs', shape=[MAX_SEQ_LEN])
    x = tf.keras.layers.Embedding(MAX_WORDS, output_dim=4, input_length=MAX_SEQ_LEN)(inputs)
    #используем LSTM вместо SimpleRNN
    x = tf.keras.layers.LSTM(units=16, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(units=8)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, name='out_layer')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    recurrent_model = tf.keras.Model(inputs=inputs, outputs=x)
    return recurrent_model


def load_data(csv_path='data/raw/spam.csv') -> tuple:
    df = pd.read_csv(csv_path)
    X = df.x.astype('str')
    Y = df.y.astype('int')
    return X, Y


def train():
    X_train, Y_train = load_data()
    X_val, Y_val = load_data('data/raw/spam_test.csv')
    tok = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
    tok.fit_on_texts(X_train)
    #преобразование текста в последовательные индексы и заполнение до макс длины
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
    model = make_model()
    model.summary()
    class_weight = {0: 0.5, 1: 3}
    #выбор функции потерь, оптимизатора и метрик
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.AdamW(3e-4),
        metrics=['accuracy', tf.keras.metrics.Precision()]
    )
    #настройка обратных вызовов,сохранение лучшей модели
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/model_7',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    #обучение модели
    model.fit(
        sequences_matrix,
        Y_train,
        batch_size=128,
        epochs=50,
        validation_data=(X_val, Y_val),
        class_weight=class_weight,
        callbacks=callbacks
    )

    def validate(model_path='models/model_7') -> tuple:
        """
        Validate model on test subset
        todo fit tokenizer on train texts,
        todo achieve >0.95 both accuracy and precision
        """
        model = tf.keras.models.load_model(model_path)
        X_train, Y_train = load_data()
        X_test, Y_test = load_data('data/raw/spam_test.csv')
        tok = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
        tok.fit_on_texts(X_train)
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN)
        loss, accuracy, precision = model.evaluate(test_sequences_matrix, Y_test)
        print(f'Test set\n  Loss: {loss:0.3f}  Accuracy: {accuracy:0.3f}, Precision: {precision:0.3f}')
        if accuracy > 0.95 and precision > 0.95:
            print("Validation successful! Accuracy and precision both above 0.95.")
        else:
            print("Validation failed! Accuracy and/or precision below 0.95.")
        return accuracy, precision

    def upload():
        """Pipeline: Upload model to S3 storage"""
        logging.info('Upload model...')

        zip_model_path = PATH_TO_MODEL + '.zip'
        shutil.make_archive(base_name=PATH_TO_MODEL, format='zip', root_dir=PATH_TO_MODEL)

        config = dotenv.dotenv_values('.env')
        ACCESS_KEY = config.get('ACCESS_KEY')
        SECRET_KEY = config.get('SECRET_KEY')

        try:
            client = boto3.client(
                's3',
                endpoint_url='https://storage.yandexcloud.net',
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY
            )
            client.upload_file(zip_model_path, BUCKET_NAME, f'{YOUR_GIT_USER}/model_7.zip')
            logging.info('Upload successful.')
        except Exception as e:
            logging.error(f'Upload failed. Error: {str(e)}')

    if name == 'main':
        parser = argparse.ArgumentParser(
            prog='src/seminar7.py',
            description='Seminar 7. SMS spam Classification with Recurrent Nets.')
        parser.add_argument('--download', action='store_true', help='Download images and extract to data/raw directory')
        parser.add_argument('--train', action='store_true', help=f'Build, train and save model to {PATH_TO_MODEL}')
        parser.add_argument('--validate', action='store_true', help='Validate model on the test subset')
        parser.add_argument('--upload', action='store_true', help='Upload model to S3 storage')
        args = parser.parse_args()

        # настройка логирования
        logging.basicConfig(level=logging.INFO)

        if args.download:
            download_data()
        if args.train:
            train()
        if args.validate:
            validate()
        if args.upload:
            upload()