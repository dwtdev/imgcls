import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.preprocessing import minmax_scale
from imgaug import augmenters as iaa

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, CenterCrop, RandomRotation

image_size = 300
input_shape = (image_size, image_size, 3)


def get_cropped_image(img_path, crop_size=300):
    '''
    Randomly select a 300x300 pixel area from the full image
    '''
    img = Image.open(img_path)
    img_height, img_width = img.size
    img = np.array(img)

    y = random.randint(0,img_height-crop_size)
    x = random.randint(0,img_width-crop_size)

    cropped_img = img[x:x+crop_size , y:y+crop_size,:]
    
    return cropped_img


def vote_on_predictions(predictions_per_run):
    final_prediction = []
    number_of_runs = len(predictions_per_run)
    for prediction_index in range(len(predictions_per_run[0])):
        prediction_list = []
        for run_index in range(number_of_runs):
            prediction_list.append(predictions_per_run[run_index][prediction_index])
        if prediction_list.count(4)==len(prediction_list):
            final_prediction.append(4)
        else:
            counts = np.bincount(prediction_list)
            final_prediction.append(int(np.argmax(counts)))
    return final_prediction 


def augmented_prediction(image_list, folder, number_of_runs=10):
    predictions_per_run = []
    for run in range(number_of_runs):
        predictions = []
        with tqdm(total=len(image_list)) as pbar:
            for image_filename in image_list:
                pbar.update(1)
                output = model.predict(np.array([get_cropped_image(folder+"/"+image_filename)]))
                predictions.append(np.argmax(output))
        predictions_per_run.append(predictions)
    return predictions_per_run


def build_model(input_shape, num_classes=1000, train=True):
    effnet_layers = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)

    if train:
        effnet_layers.load_weights("./efficientnetb3_notop.h5")

        for layer in effnet_layers.layers:
            layer.trainable = True

    dropout_dense_layer = 0.3

    model = Sequential()
    model.add(effnet_layers)
        
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(num_classes, activation="softmax"))

    if not train:
        model.load_weights('best_model.h5')
        
    model.summary()

    return model



if __name__ == "__main__":
    test_folder = './test_images/'
    test_df = pd.DataFrame(columns={"image_id","label"})
    test_df["image_id"] =  os.listdir(test_folder)
    test_df["label"] = 0  

    model = build_model(input_shape, 5, train=False)  

    predictions_per_run = augmented_prediction(test_df["image_id"], test_folder)
    test_df["label"] = vote_on_predictions(predictions_per_run)

    test_df.to_csv("test_results.csv", index=False)