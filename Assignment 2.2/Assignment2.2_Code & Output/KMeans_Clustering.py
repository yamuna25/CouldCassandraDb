# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:03:35 2016

@author: Yamuna
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot
import pylab



path = os.path.join(os.path.dirname(__file__), 'maildir')
os.chdir(path)
vectorizer = TfidfVectorizer(stop_words='english')
headers = ['Message-ID:','Mime','MIME','Content','X','Date:','Sent:','Cc:','Bcc:','From:','To:','@','X-From:','X-To:','cc:','.xls','download','____']
folder = "all_documents"
subject=[]
mail=[]
parsed_mail = {}
from pyspark import SparkContext
sc = SparkContext("local", "Parser App")

for ddir in os.listdir(path):
    #if (os.path.isdir(ddir)):
        print("dir-"+ddir)
        for sub_dir in os.listdir(path+"\\"+ddir):
            if sub_dir == folder:
                print("sub-"+sub_dir)
                for file in os.listdir(path+"\\"+ddir+"\\"+sub_dir):
                        os.chdir(path+"\\"+ddir+"\\"+sub_dir)
                        fileread=open(file, 'r')
                        data=fileread.readlines()
                        fileread.close()
                        for line in data:
                            if ("Subject:" in line) and not any(header in line for header in headers):
                                subjectset=1
                                subject.append(line)
                                key = "Subject"
                                parsed_mail[key] = line
                            if not any(header in line for header in headers) and not ("Subject:" in line):
                                mail.append(line)
                                key = "Content"
                                parsed_mail[key] = line
mail=mail+subject
X = vectorizer.fit_transform(mail)
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("Top terms per cluster in email subject:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind]
        
