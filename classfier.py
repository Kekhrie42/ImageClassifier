#Importing all the required classes below needed for the image classification.

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import subprocess
from tensorflow import keras
from keras import layers
from keras.models import Sequential


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

monFolder = '1shcZdYoZKxh1aZao5ZZMGqcVnJsRay3s'
file_list = drive.ListFile({'q':f"'{monFolder}' in parents and trashed=false"}).GetList()

print(file_list)
for index, files in enumerate(file_list):
  #print('file', index + 1, files['title'])
  #print(files)
  if(files['title'] == 'high'):
    file_high_id = files['id']
    file_list_high = drive.ListFile({'q':f"'{file_high_id}' in parents and trashed=false"}).GetList()
  elif(files['title'] == 'low'):
    file_low_id = files['id']
    file_list_low = drive.ListFile({'q':f"'{file_low_id}' in parents and trashed=false"}).GetList()
  elif(files['title'] == 'good'):
    file_good_id = files['id']
    file_list_good = drive.ListFile({'q':f"'{file_good_id}' in parents and trashed=false"}).GetList()
    
    
    
subprocess.Popen(["mkdir","high"], cwd = "/content")
subprocess.Popen(["mkdir","low"], cwd = "/content")
subprocess.Popen(["mkdir","good"], cwd = "/content")



import os
path = '/content/high'
os.chdir(path)
i = 0 

for index, file in enumerate(file_list_high):
  #print(file['title'])
  if(i<=600):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1

i = 0

path = '/content/low'
os.chdir(path)

for index, file in enumerate(file_list_low):
  #print(file['title'])
  if(i<=900):
    file.GetContentFile(file['title'])
  else:
   break
  i+=1

path = '/content/good'
os.chdir(path)

for index, file in enumerate(file_list_good):
  #print(file['title'])
  if(i<=900):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1



dataset = '/content'
data_dir = pathlib.Path(dataset)


image_count = len(list(data_dir.glob('*/*.jpg')))


os.chdir("/content")
subprocess.run("rm -r sample_data", shell =True, stderr = subprocess.PIPE)
subprocess.run("rm -r .config", shell =True, stderr = subprocess.PIPE)



batch_size = 32
img_height = 500
img_width = 500
 
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("on")
    
    
    
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 



num_classes = 3

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
              
model.summary()


epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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


subprocess.Popen(["mkdir","test"], cwd = "/content")
subprocess.Popen(["mkdir","high"], cwd = "/content/test")
subprocess.Popen(["mkdir","low"], cwd = "/content/test")
subprocess.Popen(["mkdir","good"], cwd = "/content/test")


path = '/content/test/high'
os.chdir(path)
i = 0 

for index, file in enumerate(file_list_high):
  #print(file['title'])
  if(i<=33):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1

i = 600

path = '/content/test/low'
os.chdir(path)

for index, file in enumerate(file_list_low):
  #print(file['title'])
  if(i>=600 and i<=633):
    print(file['title'])
    file.GetContentFile(file['title'])
  else:
   break
  i+=1

i = 600
path = '/content/test/good'
os.chdir(path)

for index, file in enumerate(file_list_good):
  #print(file['title'])
  if(i>=600 and i<=633):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1
  
  
from pathlib import Path
 
# assign directory
directory = '/content/test/low'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


# assign directory
directory = '/content/test/high'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    
# assign directory
directory = '/content/test/good'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    # print(file)
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
    
num_classes = 3
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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


subprocess.Popen(["mkdir","test"], cwd = "/content")
subprocess.Popen(["mkdir","high"], cwd = "/content/test")
subprocess.Popen(["mkdir","low"], cwd = "/content/test")
subprocess.Popen(["mkdir","good"], cwd = "/content/test")


path = '/content/test/high'
os.chdir(path)
i = 0 

for index, file in enumerate(file_list_high):
  #print(file['title'])
  if(i<=33):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1

i = 600

path = '/content/test/low'
os.chdir(path)

for index, file in enumerate(file_list_low):
  #print(file['title'])
  if(i>=600 and i<=633):
    print(file['title'])
    file.GetContentFile(file['title'])
  else:
   break
  i+=1

i = 600
path = '/content/test/good'
os.chdir(path)

for index, file in enumerate(file_list_good):
  #print(file['title'])
  if(i>=600 and i<=633):
    file.GetContentFile(file['title'])
  else:
    break
  i+=1
  
from pathlib import Path
 
# assign directory
directory = '/content/test/low'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
# assign directory
directory = '/content/test/high'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    

# assign directory
directory = '/content/test/good'
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    # print(file)
    path_high = file 
    img = tf.keras.utils.load_img(
    path_high, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
