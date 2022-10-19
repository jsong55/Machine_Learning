import zipfile
import os
import random
import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import tensorflow_hub as hub

def view_random_image(target_dir,target_class):
  # Setup the target directory
  target_folder = target_dir+target_class

  #get a random image path
  random_image = random.sample(os.listdir(target_folder),1)

  #read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder+"/"+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"image shape: {img.shape}") # show the image shape

  return img

def plot_loss_curves(history):
  """
  Return separate loss curves for training and validation metrics
  """
  loss=history.history["loss"]
  val_loss=history.history["val_loss"]
  accuracy=history.history["accuracy"]
  val_accuracy=history.history["val_accuracy"]
  epochs=range(len(history.history["loss"]))

  # plot loss
  plt.figure()
  plt.plot(epochs,loss,label="training_loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # plot accuracy
  plt.figure()
  plt.plot(epochs,accuracy,label="training_accuracy")
  plt.plot(epochs,val_accuracy,label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

  # create a function to import image and resize it to be able to be used in the model
def load_and_prep_image(filename, img_shape=224):
  """
  read an image from filename, turns it into a tensor and reshape it
  to (image_shape, image_shape, color_channels)
  """
  # read in the image
  img = tf.io.read_file(filename)
  # decode the read file into a tensor
  img = tf.image.decode_image(img,channels=3)
  # resize the image
  img = tf.image.resize(img,size=[img_shape,img_shape])
  # rescale the image (pixel values between 0 and 1)
  img = img/255.
  return img

def pred_and_plot(model,filename,class_names):
  """
  import and image, make a prediction with model and plots the image
  with the predicted class as the title
  """
  # import the target image and preprocess it
  img = load_and_prep_image(filename)
  # make a prediction
  pred = model.predict(tf.expand_dims(img,axis=0))

  # add in logic for multi class
  if len(pred[0])>1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:
  # get the predicted class
    pred_class = class_names[int(tf.round(pred))]


  # plot the image
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  return 0

# Create TensorBoard callback (for multiple models)
def create_tensorboard_callback(dir_name,experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Let's make a function to create a model from a url
def create_model(model_url,num_classes=10):
  """
  Takes a TensoeFlow HUB URL and creates a Keras Sequential model with it.

  Args:
  model_url(str): A Tensorflow hub feature extraction url
  num_classes(int): number of output neurons in the output layers, should
    be equal to the number of target classes

  Returns:
    An uncompiled keras sequential model with model_url as feature extractor layers and 
      dense output layer with num_class output neurons.
  """
  # Download the pretrained model and save it as a keras layer
  feature_extraction_layer = hub.KerasLayer(model_url,
                        trainable=False,
                        name="feature_extraction_layer",
                        input_shape=IMAGE_SHAPE+(3,)) # freeze the already learned patterns

  # Create our model
  model = tf.keras.Sequential([
      feature_extraction_layer,
      layers.Dense(num_classes,activation="softmax",name="output_layer")
  ])                  
  return model
