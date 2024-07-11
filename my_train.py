from neuralart.trainer import Trainer

trainer = Trainer()

batch_size = 32
img_height = 224
img_width = 224
epochs = 6

image_folder_path = "/home/test/data/train/1/dataset_jpg_train_val_test"

trainer.create_dataset_from_directory(image_folder_path, batch_size, img_height, img_width)
trainer.plot_train_batch()

# trainer.load_model_path = "/home/test/CF2/models/test_trainer/VGG16/20240702-112759-images_2560-unfreeze_2-batch_32.keras"
trainer.build_model("VGG16", trainable_layers=10, random_rotation=0.3, random_zoom=0.3, learning_rate=0.001) # initially from VGG16
trainer.run(epochs)
trainer.plot_history()
trainer.plot_confusion_matrix()
trainer.evaluate()
trainer.save_model()