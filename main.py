import os
import pickle
import numpy as np
import tensorflow
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


res_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
res_model.trainable = False

model = tensorflow.keras.Sequential([
    res_model,
    GlobalMaxPool2D()]
)

def export_features(path, model):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preprocessed_img = preprocess_input(x)
    features = model.predict(preprocessed_img).flatten()
    norm_features = features / np.linalg.norm(features)

    return norm_features

filenames = []

for file in os.listdir("images"):
    filenames.append(os.path.join("images", file))

featurelist = []

for filename in tqdm(filenames):
    featurelist.append(export_features(filename, model))

pickle.dump(featurelist, open("featurelist.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))



