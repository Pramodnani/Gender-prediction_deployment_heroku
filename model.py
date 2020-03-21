import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load  Gender data and  Gender_Prediction data
# Loading the height & weight data file
#will be using all the data for training
df = pd.read_csv('weight_height.csv')

x_train=df.iloc[:,1:]
x_train

y_train=df.iloc[:,0:1]
y_train

clf = svm.SVC()

"""### The model is now ready with all the parameters and we need to push or fit the training data into the model, this can be done by using the `fit` function."""
# Training on the dataset
clf.fit(X,Y)

model=pickle.dump(clf,open('model.pkl','wb'))
