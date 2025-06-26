import os.path
import streamlit as st
from PIL import Image
import tensorflow
import numpy as np
import cv2
import pickle
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

def feature_extraction(path, model1):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_expand)
    result = model1.predict(preprocessed_img)
    norm_result = result / norm(result)
    return norm_result
st.title("Fashion recommender system")


def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distance, indices = neighbors.kneighbors(features)
    return indices


def save_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0


uploaded_file = st.file_uploader("Drag or select an image")
filenames = pickle.load(open("filenames.pkl", "rb"))
if uploaded_file is not None:
    if save_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, use_container_width=True)

        features = feature_extraction(os.path.join("uploads", uploaded_file.name),model)


        indices = recommend(features, featurelist)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("some error occurred")




