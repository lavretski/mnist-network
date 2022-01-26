import gzip
import numpy as np
import random

normalize_pictures = 1/3000

f = gzip.open('train-images-idx3-ubyte.gz','r')
f_test = gzip.open('t10k-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 60000

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
data = data*normalize_pictures

num_images = 10000

f_test.read(16)
buf = f_test.read(image_size * image_size * num_images)
data_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data_test = data_test.reshape(num_images, image_size, image_size, 1)
data_test = data_test*normalize_pictures

f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(60000)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

f_test = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f_test.read(8)
buf = f_test.read(10000)
labels_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

#import matplotlib.pyplot as plt
#image = np.asarray(data[2000]).squeeze()
#plt.imshow(image)
