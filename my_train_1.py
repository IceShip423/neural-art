import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models, applications

# Note : we are using TensorFlow Core v2.5.0, in TensorFlow Core v2.6.0 all the data 
# augmentation layers are part of tf.keras.layers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax


'''=============================Global Variable=================================='''
USERNAME = 'pablo'
MODEL = 'VGG16'

MAIN_PATH = '/home/jupyter/' 
DATASETS_FOLDER = 'wikiart/train_val_test_True_1440/'

TRAIN_DIR = MAIN_PATH + DATASETS_FOLDER + 'train'
VAL_DIR = MAIN_PATH + DATASETS_FOLDER + 'val'
TEST_DIR = MAIN_PATH + DATASETS_FOLDER + 'test'



BATCH_SIZE = 128 # Hyper param, you can tune it
EPOCHS = 1000 # Large number, early stopping to stop training before this number
IMG_HEIGHT = 224 # VGG's dim
IMG_WIDTH = 224 # VGG's dim
NUM_CLASSES = 8 # Number of art styles


'''=============================Datasets Setup=================================='''
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_DIR,
    labels='inferred',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True)

assert len(train_ds.class_names) == NUM_CLASSES 


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=VAL_DIR,
    labels='inferred',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='categorical',
    batch_size=BATCH_SIZE)

assert len(val_ds.class_names) == NUM_CLASSES


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TEST_DIR,
    labels='inferred', # labels are generated from the directory structure
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    label_mode='categorical', # labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss
    batch_size=BATCH_SIZE)

assert len(test_ds.class_names) == NUM_CLASSES

total_images_count = (int(len(list(train_ds)))+int(len(list(val_ds)))+int(len(list(test_ds))))*BATCH_SIZE
# total_images_count = 33011 + 4123 + 4134
total_images_count



AUTOTUNE = tf.data.AUTOTUNE

# Optimizing the dataset by caching and prefetching the data
train_ds = train_ds.cache().shuffle(int(total_images_count)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)




layer_model = applications.VGG16(
    include_top=False, # We do not include VGG classification layers
    weights='imagenet', # We import VGG pre-trained on ImageNet
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
    classes=NUM_CLASSES)


layer_model.layers
layer_model.trainable = False
layer_model.layers[-2:] # Set the two last layers as trainable (including the last Conv2D)
for layer in layer_model.layers[-2:]:
    layer.trainable = True
    
    
trainable_layer_count = 0

for i in range(len(layer_model.layers)):
    if layer_model.layers[i].trainable:
        trainable_layer_count += 1
        
trainable_layer_count


data_augmentation_layers = models.Sequential([
    RandomFlip("horizontal", input_shape=(224, 224,3)),
    RandomRotation(0.3),
    RandomZoom(0.3)])


tf.keras.backend.clear_session() # Clear the layers name (in case you run multiple time the cell)

inputs = Input(shape=(224, 224, 3))

x = data_augmentation_layers(inputs) # Are not applied to validation and test dataset (made inactive, tensorflow handle it)
x = applications.vgg16.preprocess_input(x) # Does the rescaling
x = layer_model(x) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x) # Dropout to prevent overfitting

outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classification_layer')(x)

model = Model(inputs, outputs)


es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
# ! Do not use ReduceLROnPlateau if your optimizer alread handle learning rate modification !
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-8)
# You can add it to the callbacks if you want to save checkpoints
checkpoint_dir = f"{MAIN_PATH}logs/{USERNAME}/{MODEL}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-unfreeze_{trainable_layer_count}"
mcp = ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq=10,
    save_best_only=True)
recorded_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
log_dir = f"{MAIN_PATH}logs/{USERNAME}/{MODEL}/" + \
    recorded_time + \
    f"-images_{total_images_count}" + \
    f"-unfreeze_{trainable_layer_count}" + \
    f"-batch_{BATCH_SIZE}"

tsboard = TensorBoard(log_dir=log_dir)






model.compile(optimizer=Adamax(learning_rate=0.001), 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
history = model.fit(
    train_ds, 
    epochs=EPOCHS, 
    validation_data=val_ds, 
    callbacks=[es, tsboard], 
    use_multiprocessing=True)

model.save(f"{MAIN_PATH}models/{USERNAME}/{MODEL}/" + \
    recorded_time + \
    f"-images_{total_images_count}" + \
    f"-unfreeze_{trainable_layer_count}" + \
    f"-batch_{BATCH_SIZE}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = history.epoch

fig = plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
fig.savefig(f"/home/jupyter/figures/{USERNAME}/{MODEL}/{recorded_time}-images_{total_images_count}-unfreeze_{trainable_layer_count}-batch_{BATCH_SIZE}", dpi=300)

model.evaluate(test_ds, callbacks=tsboard)