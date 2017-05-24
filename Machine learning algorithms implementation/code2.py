# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 01:29:00 2016
@author: Yamuna
"""
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

#importing given dataset
in_data = pd.read_csv("satellite.csv")

# Initialising 10 fold cross validation
kf_total = cross_validation.KFold(len(in_data), n_folds=10, shuffle=False, random_state=None)

# Initialising Random Forest Classifier
rf_classifier=RandomForestClassifier(n_estimators=10, 
                          max_features='auto',
                          max_depth=None,
                          min_samples_split=2, 
                          random_state=0)

# Initialising Naive Bayes Classifier
nb_classifier = GaussianNB()

# Calculating accuracy scoreof naive bayes and Random forest for 10 folds
nb_accu_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
rf_accu_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
loop=0

# Looping for each fold 
for train_index, test_index in kf_total:
    # Training the Naive Bayes Classifier for each fold with train data and train result
    nb_classifier.fit(np.array(in_data.ix[train_index[0]:train_index[len(train_index)-1],:36]), np.array(in_data.ix[train_index[0]:train_index[len(train_index)-1],36:37]))
    # Predicting the test data using the trained Naive Bayes Classifier
    nb_predicted = nb_classifier.predict(np.array(in_data.ix[test_index[0]:test_index[len(test_index)-1],:36]))
    # Calculating accuracy of Naive Bayes Classifier for each fold and storing it in an array
    nb_accu_score[loop] = sklearn.metrics.accuracy_score(nb_predicted,np.array(in_data.ix[test_index[0]:test_index[len(test_index)-1],36:37]))
    # Training the Random Forest Classifier for each fold with train data and train result
    rf_classifier.fit(np.array(in_data.ix[train_index[0]:train_index[len(train_index)-1],:36]), (np.array(in_data.ix[train_index[0]:train_index[len(train_index)-1],36:37])).ravel())
    # Predicting the test data using the trained Random Forest Classifier
    rf_predicted=rf_classifier.predict(np.array(in_data.ix[test_index[0]:test_index[len(test_index)-1],:36]))
    # Calculating accuracy of Random Forest Classifier for each fold and storing it in an array
    rf_accu_score[loop] = sklearn.metrics.accuracy_score(rf_predicted,np.array(in_data.ix[test_index[0]:test_index[len(test_index)-1],36:37]))
    loop=loop+1

# Calculating Mean accuracy for NaiveBayes Classifier
NB_Mean_accu = np.mean(nb_accu_score) 
# Calculating Mean Accuracy for Random Forest Classifier
RF_Mean_accu = np.mean(rf_accu_score) 
# Calculating Standard deviation of NaiveBayes classifier accuracy
NB_std_dev=np.std(nb_accu_score)
# Calculating Standard deviation of Random Forest Classifier accuracy
RF_std_dev=np.std(rf_accu_score)    

from scipy import stats
# Given alpha value in question
alpha=0.05

# Calculating statistical test for Mean and standard deviation of accuracy scores in NaiveBayes and Random Forest
t, p = stats.ttest_ind_from_stats(NB_Mean_accu, NB_std_dev, 10,
                              RF_Mean_accu, RF_std_dev, 10,
                              equal_var=False)
print("ttest_ind_from_stats: t = %g  p = %g" % (t, p))

# Determinig the Statistical significance using ttest and alpha
if (p<alpha):
    print("P is less than alpha(0.05), hence there is a significant difference between the NaiveBayes and Random forest classification")
else:
    print("There is no significant difference between the NaiveBayes and Random forest classification")
