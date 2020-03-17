# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load  Gender data and  Gender_Prediction data
# Loading the height & weight data file
data1 = pd.read_csv('E:\gender_model_deply\weight_height.csv')

# Converting the index column to list
indexes = data1.index.values.tolist()
X = [[x] for x in indexes]

for x in X:
  x.extend([data1.ix[x[0],'Height'], data1.ix[x[0],'Weight']])
  
for item in X:
  item.pop(0)

"""#### Get a snapshot of the data using `head()` function."""

data1.head()

"""Seperate Gender (target) column from the data which is Gender"""

# Seperating Gender column from the earlier created list
Y = data1['Gender'].tolist()

"""#Split data into training and test sets

Split data into `training = 80%` and `testing = 20%` using `train_test_split` function:

- `random_state` is used for initializing the internal random number generator, which will decide the splitting of data into train and test indices.
"""

#x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.2, random_state=7)



#Model 
clf = svm.SVC()

"""### The model is now ready with all the parameters and we need to push or fit the training data into the model, this can be done by using the `fit` function."""

# Training on the dataset
clf.fit(X,Y)



model=pickle.dump(clf,open('E:\gender_model_deply\model.pkl','wb'))
                  
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[175,170]]))

"""### Metric to measure model accuracy"""

# Compare the accuracy and result on new data
#acc=accuracy_score(y_test,y_pred)


# Printing the accuracy score of  the model
#print(f'Accuracy: {round(acc*100,2)}%')

#import pickle 
#print(pickle.format_version)
