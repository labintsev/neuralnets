import os.path
import pickle
import shutil

import numpy as np


def load_batch(fpath, label_key="labels"):
   """Internal utility for parsing CIFAR data.

   Args:
       fpath: path the file to parse.
       label_key: key for label data in the retrieve
           dictionary.

   Returns:
       A tuple `(data, labels)`.
   """
   with open(fpath, "rb") as f:
      d = pickle.load(f, encoding="bytes")
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
         d_decoded[k.decode("utf8")] = v
      d = d_decoded
   data = d["data"]
   labels = np.array(d[label_key], dtype=int)

   data = data.reshape(data.shape[0], 3, 32, 32)
   return data, labels


def load_local_data():
   data_path = '../data'
   archive_path = os.path.join(data_path, 'cifar-10-python.tar.gz')
   cifar_path = os.path.join(data_path, 'cifar-10-batches-py')
   if not os.path.exists(cifar_path):
      shutil.unpack_archive(archive_path, data_path)

   num_train_samples = 50000

   x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
   y_train = np.empty((num_train_samples,), dtype="uint8")

   for i in range(1, 6):
      batch_path = os.path.join(cifar_path, f"data_batch_{i}")
      (
         x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000],
      ) = load_batch(batch_path)

   test_path = os.path.join(cifar_path, "test_batch")
   x_test, y_test = load_batch(test_path)
   return (x_train, y_train), (x_test, y_test)
