# Research Model -Traffic Sign Recognition Using Deep Neural Network and Skipped Layer Architecture

This notebook demonstrates a step-by-step process for building a machine learning model

## Table of Contents
1. [Clone Dataset Repository](#clone-dataset-repository)
2. [Install Keras](#install-keras)
3. [Import Libraries](#import-libraries)
4. [Load Data](#load-data)
5. [Data Preprocessing](#data-preprocessing)
6. [Data Augmentation](#data-augmentation)
7. [Normalize Data](#normalize-data)
8. [Build Model](#build-model)
9. [Compile Model](#compile-model)
10. [Model Training](#model-training)
11. [Save Model](#save-model)
12. [Test Model](#test-model)


## Clone Dataset Repository
```python
!git clone https://bitbucket.org/jadslim/german-traffic-signs
```
Explanation: Clones a repository from Bitbucket containing the German traffic signs dataset.

## List Files in the Cloned Repository
```python
!ls german-traffic-signs
```
Explanation: Lists the files present in the cloned repository

## Install Keras
```python
pip install --upgrade keras
```
Explanation: Installs the latest version of the Keras library for building neural networks.

## Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras.layers import BatchNormalization
from keras import Input, Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense
```

Explanation: Imports necessary libraries for data manipulation, visualization, and building the model.


## Set Random Seed
```python
np.random.seed(0)
```
Explanation: Sets the random seed for reproducibility.

The purpose of np.random.seed(0) is to ensure reproducibility in the code by fixing the random number generator's starting point.
Many machine learning and deep learning processes rely on randomness, such as:

- Weight initialization in neural networks

- Data shuffling during training

- Random data augmentation (e.g., rotations, flips)

- Dropout layers

If you don’t set a seed, each run of your code will produce different random numbers, leading to inconsistent results.

This ensures that every time you run your code, the "random" numbers generated will be the same.

## Load Data
```python
with open('german-traffic-signs/train.p','rb') as f:
  train_data = pickle.load(f)

with open('german-traffic-signs/valid.p','rb') as f:
  val_data = pickle.load(f)

with open('german-traffic-signs/test.p','rb') as f:
  test_data = pickle.load(f)

print(type(train_data))

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

y_train.shape = (34799, 1)
```
Explanation: Loads training, validation, and test datasets from pickle files and reshapes the training labels.

pickle.load(f) reads serialized Python objects stored in binary format.

The dataset is split into three parts:

- train.p → Training data (used to train the model)

- valid.p → Validation data (used to tune hyperparameters)

- test.p → Test data (used for final evaluation)


### Extracting Features (Images) and Labels
Purpose: Separates images (features) and their corresponding class labels (labels).

Details:

- X_train, X_val, X_test → Arrays containing traffic sign images (pixel data).

- y_train, y_val, y_test → Arrays containing class labels (e.g., 0 for speed limit 20, 1 for speed limit 30, etc.).

### Reshaping Labels
y_train.shape = (34799, 1)

Purpose: Ensures labels are in a 2D format (useful for some models or loss functions).

### Why Validation Data?

Used to monitor model performance during training and prevent overfitting.

## Print Data Shapes
```python
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(X_test.shape)
```
Explanation: Prints the shapes of the training, validation, and test datasets.

## Load Class Names
```python
data = pd.read_csv('german-traffic-signs/signnames.csv')
print(data)
```
Explanation: Loads class names from a CSV file and prints them.

### Loading the CSV File
Purpose: Reads the signnames.csv file into a Pandas DataFrame.

The file signnames.csv typically contains two columns:

- ClassId (integer): The numeric label assigned to each traffic sign (e.g., 0, 1, 2, etc.).

- SignName (string): The corresponding human-readable name (e.g., "Speed limit (20km/h)", "Stop", etc.).

signnames.csv acts as a lookup table to translate these numbers into meaningful names (useful for visualization or reporting results).

## Display Sample Image From the Training Dataset
```python
index = 0
X_train[index]
```
Explanation: Displays the first image in the training dataset.

## Plot Sample Imaget
```python
img = plt.imshow(X_train[index])
```
Explanation: Plots the first image in the training dataset.

## Print Image Label
```python
print('The image label is: ', y_train[index])
```
Explanation: Prints the label of the first image in the training dataset.

## One-Hot Encode Labels
```python
classification = ['20km/h', '30km/h', '50km/h', '60km/h', '70km/h', '80km/h', 'End of speed limit (80km/h)', '100km/h','120km/h',
                  'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road',
                  'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry','General caution',
                  'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
                  'Road work', 'Traffic signals', 'Pedestrians','Children crossing',
                  'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
                  'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left','Keep right',
                  'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric ...']

print('The image class is: ', classification[y_train[index][0]])
print(y_train)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
y_val_one_hot = to_categorical(y_val)

print(y_train_one_hot)
print('The one hot label is:', y_train_one_hot[0])
```
Explanation: One-hot encodes the labels and prints the one-hot encoded labels.

### Defining Human-Readable Class Names

Purpose: A manually defined list mapping class IDs (e.g., 0, 1, 2) to their corresponding traffic sign names.
The dataset (y_train) uses numeric labels (e.g., y_train[0] = 0), but we want to display the actual sign name (e.g., "20km/h").

### One-Hot Encoding the Labels
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
y_val_one_hot = to_categorical(y_val)

Purpose: Converts integer labels into one-hot encoded vectors (required for multi-class classification in Keras/TensorFlow).

#### Why One-Hot Encoding?

Neural networks typically require labels in one-hot format for categorical cross-entropy loss.

Example: For 43 classes, label 3 becomes a 43-dimensional vector with 1 at index 3.

#### Shape After One-Hot Encoding:

y_train_one_hot.shape = (n_samples, 43) (e.g., (34799, 43) for training data).

#### Mapping Back to Class Names:



## Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)

datagen.fit(X_train)
```
Explanation: Uses ImageDataGenerator to augment the training data.

This code sets up data augmentation for image data using Keras' ImageDataGenerator. Data augmentation artificially expands your training dataset by applying random transformations to the images, which helps improve model generalization and prevent overfitting.

Purpose: Defines a generator that applies random transformations to images during training.

#### Key Parameters:

- width_shift_range (0.1): Randomly shifts images horizontally by up to 10% of the width.

- height_shift_range (0.1): Randomly shifts images vertically by up to 10% of the height.

- zoom_range (0.2): Randomly zooms in/out by up to 20%.

- shear_range (0.1): Applies a shear transformation (slanting the image) by up to 10%.

- rotation_range (10): Rotates images by up to 10 degrees.

Note: All transformations are randomized within the specified ranges.


## Generate Augmented Batches
```python
batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)
```
Explanation: Generates batches of augmented data

batch_size=20: Each batch will contain 20 images and their corresponding labels.
next(batches) gets the first batch of augmented images and labels.
During actual training, you would use this generator in model.fit() instead of the original data.

## Normalize Data
```python
X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255
```
Explanation: Normalizes the pixel values of the images to the range [0, 1].
This is crucial because:

- Neural networks work better with small input values.
- Helps with faster convergence during training.
- Makes the optimization landscape smoother.

## Build Model
```python
input_layer=Input(shape=(32,32,3))

x1 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(input_layer)

x1 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(x1)

x1 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(x1)

x1 = BatchNormalization()(x1)

x1=MaxPooling2D(pool_size=(2, 2))(x1)

x1 = Flatten()(x1)

x2 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(input_layer)

x2 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(x2)

x2 = Conv2D(
        filters=60,
        kernel_size=[5, 5],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu )(x2)

x2 = BatchNormalization()(x2)

x2=MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Conv2D(
        filters=30,
        kernel_size=[3, 3],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu)(x2)

x2 = Conv2D(
        filters=30,
        kernel_size=[3, 3],
        kernel_initializer='he_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=0),
        padding="same",
        activation=tf.nn.relu)(x2)

x2 = BatchNormalization()(x2)

x2=MaxPooling2D(pool_size=(2, 2))(x2)

x2 = Dropout(0.5)(x2)

x2 = Flatten()(x2)

x = Concatenate(axis=-1)([x1,x2])

x = Dense(500 ,activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(43,activation='softmax')(x)

conv_model = Model(input_layer,x)
optimizer = tf.keras.optimizers.Adam(0.001)

conv_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(conv_model.summary())
```
Explanation: Builds a convolutional neural network model with multiple convolutional and max-pooling layers, followed by batch normalization, dropout, and dense layers.
#### The Proposed Model Architecture
![image](https://github.com/user-attachments/assets/23c08b4c-66ff-4fb7-9ebf-651d0cf835be)

The model uses two parallel convolutional pathways that process the input image differently before merging:

- Pathway 1 (x1): Three 5x5 convolutional layers → BatchNorm → MaxPooling → Flatten

- Pathway 2 (x2): Three 5x5 conv → BatchNorm → MaxPool → Two 3x3 conv → BatchNorm → MaxPool → Dropout → Flatten

- Merged Path: Concatenated features → Dense (500 neurons) → Dropout → Output (43 classes)

###  Component-by-Component Explanation

#### Input Layer
Defines the input tensor shape matching GTSRB image dimensions.

#### Path 1
```python
x1 = Conv2D(filters=60, kernel_size=[5,5], activation='relu')(input_layer)
x1 = Conv2D(60, [5,5], activation='relu')(x1)
x1 = Conv2D(60, [5,5], activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Flatten()(x1)
```

##### Three 5x5 Convolutional Layers:

- Extract hierarchical features (edges → shapes → structures).

- filters=60: Output depth (number of feature maps).

- kernel_initializer='he_uniform': Optimal for ReLU activations.

##### Batch Normalization: 
Stabilizes training by normalizing layer outputs.

##### Max Pooling (2x2): 
Reduces spatial dimensions by half (compression).

##### Flatten: 
Converts 3D feature maps to 1D vector for dense layers.

#### Path 2
```python
x2 = Conv2D(60, [5,5], activation='relu')(input_layer)
x2 = Conv2D(60, [5,5], activation='relu')(x2)
x2 = Conv2D(60, [5,5], activation='relu')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2,2))(x2)
x2 = Conv2D(30, [3,3], activation='relu')(x2)
x2 = Conv2D(30, [3,3], activation='relu')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D((2,2))(x2)
x2 = Dropout(0.5)(x2)
x2 = Flatten()(x2)
```
##### Initial 5x5 Convolutions: 
Same as Pathway 1 for extracting features.

##### Additional 3x3 Convolutions:
Capture finer details with smaller kernels.
Reduced filters (30) for computational efficiency.

##### Dropout (0.5): 
Randomly deactivates 50% of neurons to prevent overfitting.

### Feature Merging & Classification Head
```python
x = Concatenate(axis=-1)([x1, x2])  # Combine both pathways
x = Dense(500, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(43, activation='softmax')(x)  # 43 traffic sign classes
```

##### Concatenate: 
Merges features from both pathways (enriches representation).

##### Dense (500 neurons): 
High-level feature learning.

##### Softmax Output: 
Probability distribution over 43 classes.


## Compile Model
```python
optimizer = tf.keras.optimizers.Adam(0.001)
conv_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(conv_model.summary())
```
Explanation: Compiles the model using the Adam optimizer and categorical cross-entropy loss function and prints the model summary.

##### Adam Optimizer: 
Adaptive learning rate (lr=0.001).

###### Categorical Crossentropy: 
Standard loss for multi-class classification.

##### Accuracy Metric: 
Tracks classification performance.

 padding="same" preserves spatial dimensions.

## Model Training
```python
trained_model =conv_model.fit(X_train, y_train_one_hot,
           batch_size=512, epochs=50, validation_data =(X_val,y_val_one_hot),shuffle=1)
```
#### Parameter Explanation
- X_train - Training images(Input data (normalized 32×32 RGB images))
- y_train_one_hot - One-hot encoded labels(Target output (43 traffic sign classes))
- batch_size - Number of samples processed before updating weights
- epochs - Complete passes through the entire training dataset
- validation_data - Validation set for monitoring generalization
- shuffle - Randomizes training sample order each epoch

#### Key Training Dynamics
##### Batch Processing:

The 34,799 training samples are divided into batches of 512
68 batches per epoch (34,799/512 ≈ 68)
Weight updates occur after each batch

##### Validation Monitoring:
After each epoch, evaluates performance on the validation set
Helps detect overfitting (when validation accuracy plateaus while training accuracy improves)

##### Shuffling:
Prevents the model from learning artificial sequence patterns
Improves gradient descent quality by randomizing sample order

#### Performance Considerations
##### Batch Size (512):
Pros: Faster training (fewer weight updates)
Cons: May converge to sharper minima (potentially worse generalization)
Typical values: 32-512 (requires tuning)

##### Epochs (50):
Early stopping often used to halt when validation accuracy plateaus
For this dataset, 50 epochs is reasonable but may be excessive

##### Memory Usage:
Large batch sizes require more GPU memory
If crashes occur, reduce batch_size (e.g., to 256 or 128)

##### Callbacks
Added callbacks for better training control.

## Evaluation
```python
test_loss, test_acc = conv_model1.evaluate(X_test, y_test_one_hot)
print(f"Test accuracy: {test_acc:.4f}")
```
- Plot training vs validation accuracy/loss curves.
- Evaluate on test set.
- Generate confusion matrix to analyze class-specific performance.

#### Training Accuracy (accuracy)
The model's classification accuracy on the training set at each epoch
Shows how well the model learns the training data

#### Validation Accuracy (val_accuracy)
The model's accuracy on the held-out validation set
Indicates how well the model generalizes to unseen data

#### Key Insights You Can Gain

- Underfitting: Both lines remain low → Model isn't learning effectively
(Solution: Increase model capacity, train longer)

- Overfitting: Training accuracy keeps rising while validation accuracy plateaus/starts dropping
(Solution: Add dropout, regularization, or get more data)

- Good Fit: Both lines converge to similar high values
(Ideal scenario)

## Save Model
```python
conv_model.save('my_model.h5')
```
#### What it saves:
- Complete model architecture (layer configuration)
- All model weights (learned parameters)
- Optimizer state (allows resuming training exactly where you left off)
- Loss and metrics (everything needed to recompile the model)

#### When to use:
- When you want to save/load the entire model in one operation
- For deployment to production
- When you need to share the complete model with others

```python
conv_model.save_weights("model.h5")
```
#### What it saves:
- Only the weights (no architecture or optimizer state)
- Much smaller file than full model save

#### When to use:
- When you need to save just the learned parameters
- For transfer learning (applying weights to a similar architecture)
- When you want to version-control weights separately from architecture

### How to reuse the model
conv_model.save('traffic_sign_model.h5')

#### Later... load complete model
loaded_model = tf.keras.models.load_model('traffic_sign_model.h5')

#### Make predictions
predictions = loaded_model.predict(X_test)

## Test Model
#### Google Drive Mounting & Image Loading
```python
from google.colab import drive
drive.mount('/content/drive')
new_image = plt.imread("/content/drive/MyDrive/images/safari_1.jpg")
```

Purpose: Accesses an image stored in Google Drive
Output: Loads a test image (e.g., a traffic sign photo)

#### Image Preprocessing
```python
resized_image = resize(new_image, (32,32,3))  ##### Resize to match model input
img = plt.imshow(resized_image)              ##### Display resized image
print(resized_image.shape)                   ##### Verify shape: (32, 32, 3)
```

#### Model Prediction
```python
predictions = conv_model.predict(np.array([resized_image]))  ##### Shape: (1, 43)
```
Output: A probability distribution over 43 traffic sign classes
Example: [[0.01, 0.05, ..., 0.8]] (where 0.8 might indicate 80% confidence for class 42)


#### Prediction Analysis (Top-5 Classes)
```python
list_index = list(range(43))  # [0,1,2,...,42]

# Bubble sort to rank class probabilities (inefficient but works)
for i in range(43):
  for j in range(43):
    if predictions[0][i] > predictions[0][j]:
      list_index[i], list_index[j] = list_index[j], list_index[i]

# Print top 5 predictions
for i in range(5):
  print(classification[list_index[i]], ':', round(predictions[0][list_index[i]]*100, 2), '%')
```

#### Model Evaluation on Test Set
```python
y_pred = np.argmax(conv_model.predict(X_test), axis=-1)  ##### Predicted classes
confusion_matrix = confusion_matrix(y_test, y_pred)      ##### Compare with true labels
```

y_pred: Array of predicted class IDs (0-42) for all test images
confusion_matrix: Matrix showing prediction vs actual counts


#### Confusion Matrix Visualization
```python
plot_confusion_matrix(confusion_matrix, figsize=(43,43), show_normed=True)
```
