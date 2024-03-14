# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![image](https://github.com/aldrinlijo04/nn-classification/assets/118544279/ebcd7383-82d5-4884-948c-9b9c8916dfad)

## DESIGN STEPS

1. Import Libraries:

Import necessary libraries like pandas, numpy, etc.

2. Load and Explore Data:

Load the dataset using appropriate functions.
Explore the data to understand its structure, content, and missing values.

3. Preprocess and Clean Data:

Handle missing values (e.g., imputation, deletion).
Deal with outliers and inconsistencies.
Perform feature scaling or normalization if necessary.

4. Feature Engineering:

Encode categorical features (e.g., one-hot encoding, label encoding).
Create new features from existing ones if relevant.

5. Exploratory Data Analysis (EDA):

Visualize data distribution and relationships using various plots (e.g., histograms, scatter plots).
Gain insights into data patterns and trends.

6. Split Data into Training and Testing Sets:

Split the data into training and testing sets for model development and evaluation.

7. Build Deep Learning Model:

Design the model architecture with appropriate layers (e.g., dense, convolutional) and activation functions.
Compile the model with an optimizer and loss function.

8. Train the Model:

Train the model on the training set for a specified number of epochs.
Monitor training progress and adjust hyperparameters (e.g., learning rate, batch size) if needed.

9. Evaluate Model Performance:

Evaluate the model's performance on the testing set using various metrics (e.g., accuracy, precision, recall).
Analyze the results to assess the model's effectiveness and identify potential areas for improvement.

10. Visualize Training and Validation Performance:

Plot learning curves to visualize the model's training and validation loss and accuracy over time.
Gain insights into model convergence and potential overfitting/underfitting issues.

11. Save the Model:

Save the trained model using serialization techniques like pickle for future use or deployment.

12. Make Predictions:

Use the saved model to predict on new unseen data and generate predictions.
Interpret the predicted results and draw conclusions.

## PROGRAM

### Name: Aldrin Lijo J E
### Register Number: 212222240007

### DEPENDENCIES
```
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
```
### DATA VISUALIZATION AND CLEANING
```
data = pd.read_csv("/content/drive/MyDrive/DL exp/exp 02/customers.csv")
df = data.copy()
df.head()
df.isnull().columns
plt.figure(figsize=(10,10))
sns.histplot(df.isnull().sum())
target = df[['Segmentation']].values
df = df.drop('Segmentation',axis=1)
df.isnull().sum()
R = OrdinalEncoder()
columns_unique = df.select_dtypes(include=['object'])
columns_unique.columns
cat_cols = []
for i in columns_unique.columns:
  cat_cols.append(i)
df[cat_cols] = R.fit_transform(df[cat_cols])
df.fillna(df.mean(), inplace = True)
```
### DATA PREPROCESSING
```
O = OneHotEncoder()
target = O.fit_transform(target).toarray()
X_train,X_test,y_train,y_test = train_test_split(df,target,test_size=0.3, random_state = 49)
M = MinMaxScaler()
X_train = M.fit_transform(X_train)
X_test = M.fit(X_test)
```
### MODEL ARCHITECTURE AND TRAINING
```
model = Sequential()
model.add(Dense(10,activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(4,activation = 'softmax'))
model.summary()
model.compile('adam',loss = CategoricalCrossentropy(), metrics = ['accuracy'])
model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test))
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
```
### LOSS CURVE AND PREDICTION
```
x_test_predictions = np.argmax(model.predict(X_test), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
model.save('customer_classification_model.h5')
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train,y_train,X_test,y_test,cat_cols,df,O,L], fh)
ai_brain = load_model('customer_classification_model.h5')
with open('customer_data.pickle', 'rb') as fh:
   [X_train,y_train,X_test,y_test,cat_cols,df,O,L] = pickle.load(fh)
x_single_prediction = np.argmax(model.predict(X_test[1:2,:]), axis=1)
print(L.inverse_transform(x_single_prediction))
```
## Dataset Information
![image](https://github.com/aldrinlijo04/nn-classification/assets/118544279/6517065f-9740-47a8-9d5e-e096fc4ff372)

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![download](https://github.com/aldrinlijo04/nn-classification/assets/118544279/76e2cb3e-2370-46cf-8163-d98457b96f48)

### Classification Report
![image](https://github.com/aldrinlijo04/nn-classification/assets/118544279/c6998dd7-7ccf-475c-8856-99485436d6fb)

### Confusion Matrix
![image](https://github.com/aldrinlijo04/nn-classification/assets/118544279/fb0fca07-2590-4565-88e5-7f0259d39428)

### New Sample Data Prediction
![image](https://github.com/aldrinlijo04/nn-classification/assets/118544279/55fb2b09-41ca-431b-a558-8c24a0faa16e)


## RESULT
A neural network classification model is developed for the given dataset.
