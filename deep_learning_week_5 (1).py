#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[63]:


from sklearn.metrics import recall_score   # somehow i forgot this to load earlier.....as always.....never mind


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

# In[ ]:





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

