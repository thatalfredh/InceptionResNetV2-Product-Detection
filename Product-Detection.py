""" ====== Shopee Code League #2 Product Detection ======
 ~100k noisy training data
 42 classes
 ~ 2300 training images per class
 Evaluation criteria: Top-1 Accuracy
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import TopKCategoricalAccuracy

num_classes = 42

# Hyperparameters
learning_rate = 0.00025
epochs = 30
batch_size = 64
num_pixels = 100
freeze_proportion = 0.75
dropout_prob = 0

""" ======================== Image pre-processing ======================== """
# Split training images into train/dev sets
#import split_folders
#split_folders.ratio('dataset_train/train',output="dataset_train_splitted",seed=1337,ratio=(0.7,0.3))

# Initialize image generator for augmentation 
train_datagen = ImageDataGenerator(rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = "nearest")

val_datagen = ImageDataGenerator(rescale = 1./255)

# Pass img folders into generator for augmentation
training_set = train_datagen.flow_from_directory('dataset_train_splitted/train',
                                                 target_size = (num_pixels, num_pixels),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

val_set = val_datagen.flow_from_directory('dataset_train_splitted/val',
                                            target_size = (num_pixels, num_pixels),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

""" =============== Model building using InceptionResNetV2 =============== """
# Load pre-trained network
model = Sequential()
pre_trained_conv = InceptionResNetV2(include_top=False,weights='imagenet',input_tensor=None,input_shape=(num_pixels,num_pixels,3))
model.add(pre_trained_conv)
model.add(Flatten())
model.add(Dense(100, activation="relu", name="Dense_1"))
model.add(Dense(100, activation="relu", name="Dense_2"))

# Attach classifier
model.add(Dropout(dropout_prob,name="Dropout-Regularization"))
model.add(Dense(num_classes,activation="softmax", name="Output"))
model.summary()

# Freezing pre-trained layer
for cnn_layer in model.layers[0].layers[:int(len(model.layers[0].layers)*freeze_proportion)]:
    cnn_layer.trainable = False
if int(freeze_proportion) == 1:     
    model.layers[0].trainable = False

#for cnn_layer in model.layers[0].layers[int(len(model.layers[0].layers)*freeze_proportion):]:
#    cnn_layer.trainable = True

# Check freezed layers
for layer in model.layers:
    if (layer.name == "inception_resnet_v2"):
        print("InceptionResNetV2 Layers :")
        for l in layer.layers:
            print(l.name,l.trainable)
    print("="*20)
    print(layer.name, layer.trainable)
    
model.summary()
model.compile(optimizer = Adam(learning_rate=learning_rate), 
              loss = 'categorical_crossentropy', 
              metrics = [TopKCategoricalAccuracy(k=1)])

# Train the model
model.fit_generator(training_set,
                    steps_per_epoch = len(training_set.filenames)//batch_size,
                    epochs = epochs,
                    validation_data = val_set,
                    validation_steps = len(val_set.filenames)//batch_size)
"""
# save weights only
model.save_weights('Final_Model_weights.h5')

# save architecture only
model_architecture = model.to_json()
with open('Final_Model_architecture.json','w') as json_file:
    json_file.write(model_architecture)
"""

""" ============================= Prediction ============================= """
"""
# Load model architecture and weights
from keras.models import model_from_json
with open('Final_Model_architecture.json','r') as json_file:
    architecture = json_file.read()
    
model = model_from_json(architecture)
model.load_weights('Final_Model_weights.h5')
"""

import numpy as np
import pandas as pd
from keras.preprocessing import image

test_set = pd.read_csv('test.csv')
filepath = 'dataset_test/images/'

prediction_set = [] # it is easier to convert using list first
for img in test_set['filename']:
    test_image = image.load_img(str(filepath + img), target_size = (num_pixels, num_pixels))
    prediction_set.append(np.array(test_image)/255)

prediction_set = np.asarray(prediction_set)
prediction_set.shape
predictions = model.predict(prediction_set)

categories = np.argmax(predictions, axis=1)
label = training_set.class_indices 
label = dict((v,k) for k,v in label.items())
predictions = [label[k] for k in categories]

results=pd.DataFrame({"filename":test_set['filename'],
                      "category":predictions})
results.to_csv("final_result.csv",index=False)
