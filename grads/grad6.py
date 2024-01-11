"""Run from project root, as we need .env file and download models"""

import shutil
import boto3
import dotenv
import tensorflow as tf

PATH_TO_S3_MODEL = 'models/model_6_s3'
PATH_TO_DATA = 'data/raw/cats_dogs_test'
config = dotenv.dotenv_values('.env')
ACCESS_KEY = config['ACCESS_KEY']
SECRET_KEY = config['SECRET_KEY']

client = boto3.client(
    's3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_TO_DATA,
    seed=1337,
    image_size=(180, 180),
    batch_size=16,
)


def test_model_s3(git_user):
    try:
        client.download_file('neuralnets2023',
                         f'{git_user}/model_6.zip',
                         PATH_TO_S3_MODEL + '.zip')
    except:
        return 0
    shutil.unpack_archive(PATH_TO_S3_MODEL + '.zip', PATH_TO_S3_MODEL)
    model = tf.keras.models.load_model(PATH_TO_S3_MODEL)
    try:
        score = model.evaluate(test_ds)
    except ValueError:
        return 4
    if score[1] > 0.8:
        return 5
    else:
        return 3


if __name__ == '__main__':
    with open('grads/repos.csv') as f:
        users = f.readlines()
    scores = []
    for user in users:
        git_name = user.split('.com/')[1].split('/')[0]
        name = user.split(',')[0]
        scores.append(f'{name},{test_model_s3(git_name)}\n')
    scores.append('name,score6\n')
    with open('grads/scores_6.csv', 'w') as f:
        f.writelines(sorted(scores))
