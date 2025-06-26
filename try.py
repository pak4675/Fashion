import pickle

import tensorflow
import numpy as np
import cv2
from numpy.linalg import norm
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors



featurelist = pickle.load(open("featurelist.pkl", "rb"))
featurelist = np.array(featurelist)


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])


img = image.load_img('sample/1537.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_expand = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(img_expand)
result = model.predict(preprocessed_img)
norm_result = result /norm(result)

neighbors = NearestNeighbors(n_neighbors = 6, algorithm='brute')
neighbors.fit(featurelist)

distance, indices = neighbors.kneighbors(norm_result)
filenames = pickle.load(open("filenames.pkl", "rb"))
for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow("image", cv2.resize(temp_img, (224, 224)))
    cv2.waitKey(0)



