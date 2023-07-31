import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#tf._version_

import os
import zipfile

'''local_zip = 'Classification/training_set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('Classification/training_set')
local_zip = 'Classification/validation_set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('Classification/validation_set')
zip_ref.close()'''

# The contents of the .zip are extracted to the base directory /dataset,
# which in turn each contain cats and dogs subdirectories.
# Let's now define the 4 directories

train_dogs_dir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/images_1/train/aphids')

train_cats_dir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/images_1/train/army_worm')

#mites_dir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/images_1/train/army_worm')




validation_dogs_dir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/images_1/validation/aphids')

validation_cats_dir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/images_1/validation/army_worm')

#Let's view the file (image) labels of our dataset

train_dogs_names = os.listdir(train_dogs_dir)
print(train_dogs_names[:10])

train_cats_names = os.listdir(train_cats_dir)
print(train_cats_names[:10])

validation_dogs_hames = os.listdir(validation_dogs_dir)
print(validation_dogs_hames[:10])

validation_cats_names = os.listdir(validation_cats_dir)
print(validation_cats_names[:10])

# Now we can View the number of Cats and Dogs images in the dataset

print('total training dogs images:', len(os.listdir(train_dogs_dir)))
print('total training cats images:', len(os.listdir(train_cats_dir)))
print('total validation dogs images:', len(os.listdir(validation_dogs_dir)))
print('total validation cats images:', len(os.listdir(validation_cats_dir)))

# Let's now view sample of pictures from our dataset!

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Now, we will display a batch of 8 dogs and 8 cats pictures:
# Setting up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_dogs_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dogs_names[pic_index-8:pic_index]]
next_cats_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cats_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_dogs_pix+next_cats_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# Preprocessing the training set and applying data augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range =0.2 , ### Choose a shear_range
                                   zoom_range = 0.2, ### Choose a zoom range
                                   horizontal_flip = True) ### Assign the Horizontal flip 
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Colab_Notebooks/images_1/train',
                                                 target_size = (64, 64),
                                                 batch_size =32 , ### Choose the batch size
                                                 class_mode = 'binary')

# ---------------------------------------------   ------------------------------------------------

cnn = tf.keras.models.Sequential()

# Note the input shape is the desired size of the image 64*64 with 3 bytes color
# Create the first Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Create a Pooling Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Create the second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))


# Add another Pooling Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten the results to feed into the CNN
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Convolutional Neural Network with 128 neuron hidden layer
cnn.add(tf.keras.layers.Dense(units=128, activation= 'relu'  )) ### Choose Activation Function

# Creating the Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation= 'sigmoid'  )) ### Choose Activation Function

cnn.summary()

# We will train our model with the binary_crossentropy loss,
# because it's a binary classification problem and our final activation is a sigmoid.
# We will use the adam optimizer.
# During training, we will want to monitor classification accuracy.
cnn.compile(optimizer ='adam' , loss = 'binary_crossentropy', metrics = ['accuracy']) ### Choose Optimizer

#Training our CNN on the training set and evaluating it on the test set
cnn.fit(x = training_set, validation_data = validation_set, epochs = 25)

# ------------------------------------    -------------------------------------



import numpy as np
from tensorflow.keras.preprocessing import image
#test_image = image.load_img('/content/drive/MyDrive/cnn_dataset/test_aphid.jpg', target_size = (64, 64)) ### TRY Your Own Image!
test_image = image.load_img('/content/drive/MyDrive/cnn_dataset/army_worm.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
  prediction = 'army_worm'
else:
  prediction ='aphids' 

print(prediction)
