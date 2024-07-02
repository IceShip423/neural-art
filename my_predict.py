import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras import models

MODEL_PATH="/home/test/CF2/models/test_trainer/VGG16/20240702-112759-images_2560-unfreeze_2-batch_32.keras"
IMAGE_HEIGHT=224
IMAGE_WIDTH=224
CLASS_NAMES = ['cartoon', 'others']

IMAGE_DIR="/home/test/data/laion400m-data"
OUTPUT_DIR="/home/test/data/dataset_predict_output"



class Predictor():
    def __init__(self, img_height=224,img_width=224,class_names=['cartoon', 'others']):
        self.model = None
        self.model_path = None
        self.img_height = img_height
        self.img_width = img_width
        self.class_names = class_names

    def decode_image(self,image):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(tf.constant(image), channels=3)
        return tf.image.resize(img,[self.img_height, self.img_width])

    def load_model(self,model_path):
        self.model_path = model_path
        self.model = models.load_model(model_path)

    def get_prediction(self,image):
        assert self.model, "Please load a model using the load_model() method"
        return self.model.predict(tf.expand_dims(self.decode_image(image), axis=0))


predictor = Predictor(IMAGE_HEIGHT, IMAGE_WIDTH,CLASS_NAMES)
predictor.load_model(MODEL_PATH)

images = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f)) and f.endswith(".jpg")]
os.makedirs(os.path.join(OUTPUT_DIR,"cartoon"), exist_ok=True)
for img in images:
    if not img.endswith(".jpg"):
        continue
    image = tf.io.read_file(os.path.join(IMAGE_DIR, img))
    result = predictor.get_prediction(image)
    # if the model predicts cartoon, move to new folder. Otherwise delete the image.
    if np.argmax(result[0]) == 0:
        shutil.move(os.path.join(IMAGE_DIR, img), os.path.join(OUTPUT_DIR, "cartoon", img))
    else:
        os.remove(os.path.join(IMAGE_DIR, img))