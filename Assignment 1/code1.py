# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:47:58 2016
@author: Yamuna
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm

# Read dataset
in_data = pd.read_csv("satellite.csv")

# Cleaning dataset
data_clean = in_data.dropna()
data_clean.dtypes
data_clean.describe()

# Defining predictors
predictors = data_clean[['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ']]

# Defining target data
target = data_clean.AK

# Dividing train and test data
train_data, test_data, target_train, target_test = train_test_split(predictors,target,test_size=.3)

# Shaping data
train_data.shape
test_data.shape
target_train.shape
target_test.shape

# Initialising Decision tree classifier
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train_data,target_train)

# Predicting test and train data
test_predictions=classifier.predict(test_data)
train_predictions=classifier.predict(train_data)

#Train data confusion matrix and accuracy score
train_confu_mat = skm.confusion_matrix(target_train, train_predictions)
train_accu_score = skm.accuracy_score(target_train, train_predictions)

#Test data confusion matrix and accuracy score
test_confu_mat = skm.confusion_matrix(target_test, test_predictions)
test_accu_score = skm.accuracy_score(target_test, test_predictions)

#Visualising Decision Tree
from sklearn import tree
from io import BytesIO as StringIO
from IPython.display import Image
import pydotplus

out = StringIO()
tree.export_graphviz(classifier, out_file = out)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
graph.write_pdf("DTree.pdf")
