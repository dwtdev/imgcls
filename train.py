"""
Train image classification using EfficientNet
"""
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


batch_size = 16
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
    # print(cropped_img.shape)
    
    return cropped_img


augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.2,
                 iaa.GaussianBlur(sigma=(0,2))),
    iaa.Sometimes(0.2,
                 iaa.Grayscale(alpha=(0.0, 1.0))),
    iaa.Sometimes(0.2,
                 iaa.GammaContrast((0.5, 2.0))),
    iaa.Affine(scale={"x": (1, 1.2), "y":(1, 1.2)})
])

def custom_generator(image_path_list, groundtruth_list, folder, batch_size=16, training_mode=True):
    
    while True:
        for start in range(0, len(image_path_list), batch_size):
            X_batch = []
            Y_batch = []
            end = min(start + batch_size, training_item_count)

            image_list = [get_cropped_image(folder+"/"+image_path) for image_path in image_path_list[start:end]]
    
            if training_mode:
                image_list = augmentation.augment_images(images=image_list)

            X_batch = np.array(image_list)
            Y_batch = tf.keras.utils.to_categorical(np.array(groundtruth_list[start:end]), 5) 

            # print(X_batch.shape, Y_batch.shape)
            yield X_batch, Y_batch


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

    training_folder = './train_images'

    samples_df = pd.read_csv("train.csv")
    samples_df = shuffle(samples_df, random_state=42)
    samples_df["label"] = samples_df["label"].astype("str")

    training_percentage = 0.9
    training_item_count = int(len(samples_df)*training_percentage)
    validation_item_count = len(samples_df)-int(len(samples_df)*training_percentage)
    # training_df = samples_df[:training_item_count]
    # validation_df = samples_df[training_item_count:]
    training_df = samples_df[:128]
    validation_df = samples_df[128:256]

    classes_to_predict = sorted(training_df.label.unique())
    # print(len(classes_to_predict))
    model = build_model(input_shape, len(classes_to_predict), train=True)

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),
                EarlyStopping(monitor='val_loss', patience=3),
                ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))


    class_weights = class_weight.compute_class_weight("balanced", classes_to_predict, training_df.label.values)
    class_weights_dict = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}


    history = model.fit_generator(custom_generator(training_df["image_id"], training_df["label"], training_folder, batch_size=batch_size, training_mode=True),
                    steps_per_epoch = int(len(training_df)/batch_size),
                    epochs = 20, 
                    validation_data=custom_generator(validation_df["image_id"], validation_df["label"], training_folder, batch_size=batch_size),
                    validation_steps=int(len(validation_df)/batch_size),
                    class_weight=class_weights_dict,
                    callbacks=callbacks)