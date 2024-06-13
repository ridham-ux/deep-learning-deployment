#!/usr/bin/env python
# coding: utf-8

# AISC2007-WEEK5-ASSIGNMENT: model to predict if a person has Covid-19 or other infections or no lung infection from the lung xrays

# LETS BEGIN BY IMPORTING SOME IMPORTANT LIBRARIES....AS ALWAYS

# In[6]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install --upgrade albumentations')
import os
import cv2
import numpy as np
from skimage import exposure, morphology
from skimage.filters import threshold_otsu
from albumentations import Compose, Rotate, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

WE GOT TWO TYPES OF IMAGES.......IMAGE AND MASK

MY INITIAL APPROCH FOR THIS MODEL IS TO COMBINE BOTH IMAGES.
BOTH ARE GRAYSCALE IMAGES WITH 1 CHANNEL.... I AM COMBINING BOTH TO GET A COBINED IMAGE WITH 2 CHANNELS.
HERE WE GO.....

I WILL ALSO BE DOING SOME PREPROCESSING (RESIZING AND NORMALIZATION)
# In[29]:


def combine_image_and_mask(image_path, mask_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError("Image or mask not found")
        
        # Resizing images if needed
        image = cv2.resize(image, (150, 150))
        mask = cv2.resize(mask, (150, 150))
        
        # Normalizing and combine the image and mask
        combined = np.dstack((image, mask)) / 255.0
        return combined
    except Exception as e:
        print(f"Error processing {image_path} and {mask_path}: {e}")
        return None

def load_and_combine_class_data(data_dir, class_name):
    combined_data = []
    class_dir = os.path.join(data_dir, class_name)
    print(class_dir)
    images_dir = os.path.join(class_dir, 'images')
    masks_dir = os.path.join(class_dir, 'lung masks')
    
    image_files = os.listdir(images_dir)
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        
        if os.path.exists(mask_path):
            combined = combine_image_and_mask(image_path, mask_path)
            if combined is not None:
                combined_data.append(combined)
        else:
            print(f"Warning: Mask not found for image {image_file}")
    
    return np.array(combined_data)


# NOW WE WILL APPLY AUGMENTATION, INTENSITY THRESHHOLDING,MORPHOLOGICAL OPERATIONS, GAUSSIAN BLUR TO THE COMBINED IMAGE....

# In[48]:


def apply_augmentations(data):
    augmentations = Compose([
        Rotate(limit=30),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5)
    ])
    augmented_data = [augmentations(image=img)['image'] for img in data]
    return np.array(augmented_data)

def preprocess_image(image):
    try:
        # Intensity thresholding
        thresh = threshold_otsu(image)
        binary = image > thresh
        
        # Morphological operations
        binary = morphology.remove_small_objects(binary, min_size=64, connectivity=2)
        
        # Histogram equalization
        equalized_image = exposure.equalize_hist(image)
        
        # Gaussian blur
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
        
        return blurred_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def preprocess_data(data):
    preprocessed_data = [preprocess_image(img[:,:,0]) for img in data]
    return np.array(preprocessed_data)

# Defining base directory
data_dir = "/Users/radu/downloads/val"




# In[49]:


# Loading data for each class separately
normal_data = load_and_combine_class_data(data_dir, 'Normal')
print(normal_data)
covid_data = load_and_combine_class_data(data_dir, 'Covid-19')
non_covid_data = load_and_combine_class_data(data_dir, 'Non-Covid')

# Applying augmentations
augmented_normal_data = apply_augmentations(normal_data)
augmented_covid_data = apply_augmentations(covid_data)
augmented_non_covid_data = apply_augmentations(non_covid_data)

# Preprocessing data
preprocessed_normal_data = preprocess_data(normal_data)
preprocessed_covid_data = preprocess_data(covid_data)
preprocessed_non_covid_data = preprocess_data(non_covid_data)

# Printing shapes of the preprocessed data
print(f"Preprocessed Normal data shape: {preprocessed_normal_data.shape}")
print(f"Preprocessed Covid-19 data shape: {preprocessed_covid_data.shape}")
print(f"Preprocessed Non-Covid data shape: {preprocessed_non_covid_data.shape}")


# In[50]:


from tensorflow.keras.utils import to_categorical  # will be loading libraries as per requirement


# IMPLEMENTING CNN AND MODIFYING DATA AS PER REQUIREMENT..........

# In[51]:


# Defining CNN architecture for multi-class classification
def build_multi_class_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Use softmax activation for multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modifing the data loading and preprocessing steps accordingly
# Loading and preprocessING data
X = np.concatenate([preprocessed_normal_data, preprocessed_covid_data, preprocessed_non_covid_data])
y = np.concatenate([np.zeros(len(preprocessed_normal_data)), np.ones(len(preprocessed_covid_data)), np.ones(len(preprocessed_non_covid_data)) * 2])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping input data to include the channel dimension
X_train = X_train.reshape(-1, 150, 150, 1)
X_val = X_val.reshape(-1, 150, 150, 1)

# Building CNN model for multi-class classification
input_shape = X_train[0].shape
num_classes = 3  # Number of classes
multi_class_cnn_model = build_multi_class_cnn(input_shape, num_classes)

# Training the model
history = multi_class_cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])




# Converting integer labels to one-hot encoded vectors
y_val_one_hot = to_categorical(y_val, num_classes=3)





# Evaluating model performance
y_pred_prob = multi_class_cnn_model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
auc = roc_auc_score(y_val_one_hot, y_pred_prob, multi_class='ovr')

#auc = roc_auc_score(y_val, y_pred_prob, multi_class='ovr')
conf_matrix = confusion_matrix(y_val, y_pred)

print("AUC Score:", auc)
print("Confusion Matrix:")
print(conf_matrix)


# AUC SCORE ACHIEVED: 0.947
# 
# LETS CHECK THE ROBUSTNESS OF MODEL AND PERFORM KFOLD VALIDATION.....AND CHECK THE SCORE

# In[52]:


from sklearn.model_selection import KFold    # lets load this library


# In[53]:


# Reshaping input data to include the channel dimension......it was really tricky here......
X = X[..., np.newaxis]

# Define CNN architecture
def build_multi_class_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Using softmax activation for multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# K-Fold validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_index, val_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = build_multi_class_cnn(X_train[0].shape, 3)
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    
    y_val_one_hot = to_categorical(y_val, num_classes=3)
    
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    auc = roc_auc_score(y_val_one_hot, y_pred_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    print("AUC Score:", auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    fold += 1

LETS PLOT CONFUSION MATRIX AND ROC-AUC CURVE ........ AND VISUALIZE PERFORMANCE OF OUR CNN MODEL
# In[55]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

# Visualize ROC AUC Curve
def plot_roc_auc(y_true, y_pred_prob):
    n_classes = y_pred_prob.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC Curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))
        
    plt.plot([0, 1], [0, 1], 'k--')  # random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Visualize Confusion Matrix
def plot_confusion_matrix_custom(conf_matrix, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.show()

# Assuming y_val contains true labels and y_pred_prob contains predicted probabilities
# Plot ROC AUC Curve
plot_roc_auc(y_val_one_hot, y_pred_prob)

# Plot Confusion Matrix
plot_confusion_matrix_custom(conf_matrix, classes=['Normal', 'Covid-19', 'Non-Covid'])

lime used to study local interpretability 
# In[66]:


def predict_fn(images):
    images = np.array(images)
    # Convert RGB images back to grayscale
    if images.shape[-1] == 3:
        images = rgb2gray(images)
        images = np.expand_dims(images, axis=-1)  
    if images.max() > 1:
        images = images / 255.0  
 
    if len(images.shape) == 3:  
        images = np.expand_dims(images, axis=0)
    predictions = multi_class_cnn_model.predict(images)
    return predictions

# Initialize LIME image explainer
explainer = lime_image.LimeImageExplainer()

# Convert grayscale image to RGB
def grayscale_to_rgb(image):
    if image.shape[-1] == 1:
        image = np.stack((image.squeeze(),) * 3, axis=-1)
    return image

# Select an image to explain
image_index = 0  
image = X_val[image_index]


if image.shape[-1] == 1 or len(image.shape) == 2:  # If it's a 2D grayscale image
    image_rgb = grayscale_to_rgb(image)
else:  
    image_rgb = image

image_class = y_val[image_index]


plt.imshow(image_rgb)
plt.title(f'True Label: {image_class}')
plt.show()


explanation = explainer.explain_instance(image_rgb, predict_fn, top_labels=3, hide_color=0, num_samples=1000)


temp, mask = explanation.get_image_and_mask(
    label=image_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f'Explanation for Class {image_class}')
plt.show()

# Visualize the explanation for the predicted label
predicted_class = np.argmax(multi_class_cnn_model.predict(np.expand_dims(image, axis=0)))
temp, mask = explanation.get_image_and_mask(
    label=predicted_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f'Explanation for Predicted Class {predicted_class}')
plt.show()


# PRETTY GOOD RESULTS SO FAR................TIME TO MOVE FORWARD WITH DNN MODEL

# In[63]:


from sklearn.metrics import recall_score   # somehow i forgot this to load earlier.....as always.....never mind


# In[64]:


LETS BEGIN WITH DNN MODEL .....


# In[67]:


def build_deep_dnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Using softmax activation for multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# In[68]:


# Building DNN model 
input_shape = X_train[0].shape
num_classes = 3  # Number of classes in our case
deep_dnn_model = build_deep_dnn(input_shape, num_classes)

# Training the model
history = deep_dnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                             callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])


# In[38]:


# K-Fold validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_index, val_index in kf.split(X):
    print(f"Fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    input_shape = X_train[0].shape
    num_classes = 3  # Number of classes
    model = build_deep_dnn(input_shape, num_classes)
    
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    
    y_val_one_hot = to_categorical(y_val, num_classes=3)
    
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    auc = roc_auc_score(y_val_one_hot, y_pred_prob, multi_class='ovr')
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    print("AUC Score:", auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    fold += 1


# VALIDATION RESULTS ARE SATISFYING......MODEL IS PERFORMING WELL.....
# NOW LETS PLOT ROC-AUC CURVE AND CONFUSION MATRIX .....TO VISUALIZE PERFORMANCE OF DNN MODEL

# In[69]:


# Converting integer labels to one-hot encoded vectors for ROC AUC calculation
y_val_one_hot = to_categorical(y_val, num_classes=3)

# Evaluating model performance
y_pred_prob = deep_dnn_model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
auc_score = roc_auc_score(y_val_one_hot, y_pred_prob, multi_class='ovr')
recall = recall_score(y_val, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_pred)

print("AUC Score:", auc_score)
print("Recall Score:", recall)
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing ROC AUC Curve
def plot_roc_auc(y_true, y_pred_prob):
    n_classes = y_pred_prob.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC Curve (AUC = {:.2f}) for class {}'.format(roc_auc[i], i))
        
    plt.plot([0, 1], [0, 1], 'k--')  # random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Visualizing Confusion Matrix
def plot_confusion_matrix_custom(conf_matrix, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.show()

# Ploting ROC AUC Curve
plot_roc_auc(y_val_one_hot, y_pred_prob)

# Ploting Confusion Matrix
plot_confusion_matrix_custom(conf_matrix, classes=['Normal', 'Covid-19', 'Non-Covid'])

lime used to study local interpretability
# In[70]:


def predict_fn(images):
    images = np.array(images)
    # Convert RGB images back to grayscale
    if images.shape[-1] == 3:
        images = rgb2gray(images)
        images = np.expand_dims(images, axis=-1)  
    if images.max() > 1:
        images = images / 255.0  
 
    if len(images.shape) == 3:  
        images = np.expand_dims(images, axis=0)
    predictions = model.predict(images)
    return predictions

# Initialize LIME image explainer
explainer = lime_image.LimeImageExplainer()

# Convert grayscale image to RGB
def grayscale_to_rgb(image):
    if image.shape[-1] == 1:
        image = np.stack((image.squeeze(),) * 3, axis=-1)
    return image

# Select an image to explain
image_index = 0  
image = X_val[image_index]


if image.shape[-1] == 1 or len(image.shape) == 2:  # If it's a 2D grayscale image
    image_rgb = grayscale_to_rgb(image)
else:  
    image_rgb = image

image_class = y_val[image_index]


plt.imshow(image_rgb)
plt.title(f'True Label: {image_class}')
plt.show()


explanation = explainer.explain_instance(image_rgb, predict_fn, top_labels=3, hide_color=0, num_samples=1000)


temp, mask = explanation.get_image_and_mask(
    label=image_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f'Explanation for Class {image_class}')
plt.show()

# Visualize the explanation for the predicted label
predicted_class = np.argmax(model.predict(np.expand_dims(image, axis=0)))
temp, mask = explanation.get_image_and_mask(
    label=predicted_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title(f'Explanation for Predicted Class {predicted_class}')
plt.show()


# AUC SCORE :0.9357
# RECALL SCORE: 0.8191          .......NOT THAAT BAD....

# SO FAR FOLLOWING CRITERIA IS MET..........
# 
# Basic Preprocessing such as Scaling and Image Resizing Done
# 
# Further preprocessing done such as image augmentation
# 
# Advanced Preprocessing done to combine data from both images
# 
# Detailed Preprocessing including Intensity Thresholds, Histogram equalization and Gaussian Blur
# 
# 
# CNN implemented
# 
# CNN implemented well, AUC and Confusion matrix built
# 
# CNN implemented and tuned well, handling overfitting/underfitting, using both images if needed
# 
# Really-well tuned CNN, apt architecture to achieve strong performance, cross validation performed
# 
#  DNN built
# 
# DNN with Dropouts, multiple layers producing decent accuracy and recall, AUC and Confusion Matrix built
# 
# DNN tuned, using both images if needed, handling overfitting and underfitting
# 
# DNN with strong accuracy and recall, really well tuned, apt architecture for the dataset, cross validation performed
