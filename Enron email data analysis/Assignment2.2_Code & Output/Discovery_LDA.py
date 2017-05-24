# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 01:10:56 2016

@author: Yamuna
"""
from pyspark import SparkContext
from Email_Class import EmailTemplate
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pyspark.sql import SQLContext
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from sklearn.feature_extraction.text import TfidfVectorizer
# create English stop words list
en_stop = get_stop_words('en')
from nltk.corpus import stopwords
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


if __name__ == "__main__":    
    mail_sub=[]
    def fromFunc(data):
        emailid="bass-e"
        emailid=emailid.split("-")
        fromid = []
        filename,contents=data
        contents = contents.split("\n")
        count=len(contents)
        for i in range (0,count):
            if not (len(contents[i])==0):
                if "From:" in contents[i]:
                    splitcontent=contents[i].split("From:")
                    fromid.append(splitcontent)        
                    if any(emailid[0] in id for id in splitcontent):
                        from_mail_id=contents[i]
                    else:
                        from_mail_id="not there"
        return(fromid,from_mail_id)

    def toFunc(data):
        to_id = []
        bo=[]
        subject=[]
        emailid="bass-e"
        emailid=emailid.split("-")
        filename,contents=data
        contents = contents.split("\n")
        count=len(contents)
        for i in range (0,count):
            if not (len(contents[i])==0):
                if ("@" in contents[i]) and ('To:' in contents[i]):
                    splitcontent=contents[i].rstrip('').split("To:")
                    to_id=splitcontent[1]
                elif ("@" in contents[i]) and ("From:" in contents[i]):
                    splitcontent=contents[i].rstrip('').split("From:")
                    from_id=splitcontent[1]
                elif ("Subject:" in contents[i]):
                    subject.append(contents[i])
                else:
                    bo.append(contents[i])     
        body=bo
        countbody=" ".join(bo)
        strlen=re.split(r'[^0-9A-Za-z]+',countbody)
        wordcount=len(strlen)
        filename=filename.split('/')[8].replace('_', ' ')
        if (len(to_id)==0):
            to_id="undisclosed recepients"
        return EmailTemplate(filename,from_id,to_id,subject,body, wordcount)
    
    def LDA(data):
        bo1=[]
        subject1=[]
        to_id1=[]
        filename1,contents1=data
        contents1 = contents1.split("\n")
        count=len(contents1)
        for i in range (0,count):
            if not (len(contents1[i])==0):
                if ("@" in contents1[i]) and ('To:' in contents1[i]):
                    splitcontent=contents1[i].rstrip('').split("To:")
                    #to_id1=splitcontent[1]
                elif ("@" in contents1[i]) and ("From:" in contents1[i]):
                    splitcontent=contents1[i].rstrip('').split("From:")
                    from_id1=splitcontent[1]
                elif ("Subject:" in contents1[i]):
                    subject1.append(contents1[i])
                else:
                    bo1.append(contents1[i])     
        body1=bo1+subject1
        
        filename1=filename1.split('/')[8].replace('_', ' ')
        if (len(to_id1)==0):
            to_id1="undisclosed recepients"
# list for tokenized documents in loop
        texts = []
    # loop through document list
        
        for i in body1:    
    # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
            texts.append(stemmed_tokens)
# turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        #print(corpus[0])
# generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
        print("User: "+filename1)
        print(ldamodel.print_topics(num_topics=3, num_words=3))
        #ldamodel.saveAsTextFile=("LDA")
        return (filename1, subject1, ldamodel.print_topics(num_topics=3, num_words=3))

    def KMeansCluster(data):
        bo1=[]
        subject1=[]
        to_id1=[]
        filename1,contents1=data
        contents1 = contents1.split("\n")
        count=len(contents1)
        for i in range (0,count):
            if not (len(contents1[i])==0):
                if ("@" in contents1[i]) and ('To:' in contents1[i]):
                    splitcontent=contents1[i].rstrip('').split("To:")
                    #to_id1=splitcontent[1]
                elif ("@" in contents1[i]) and ("From:" in contents1[i]):
                    splitcontent=contents1[i].rstrip('').split("From:")
                    #from_id1=splitcontent[1]
                elif ("Subject:" in contents1[i]):
                    subject1.append(contents1[i])
                else:
                    bo1.append(contents1[i])     
        body1=bo1+subject1
        
        filename1=filename1.split('/')[8].replace('_', ' ')
        if (len(to_id1)==0):
            to_id1="undisclosed recepients"
        vectorizer = TfidfVectorizer(stopwords.words('english'))
        X = vectorizer.fit_transform(body1)
        true_k = 2
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        print("Top terms per cluster in email subject:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print "Cluster %d:" % i,
            for ind in order_centroids[i, :10]:
                print '  %s' % terms[ind]
        return (filename1, terms[ind])
        
    sc = SparkContext("local", "Parser App")
    baseRdd = sc.wholeTextFiles(".\\maildir\\*\\all_documents\\*")

    MainRdd=baseRdd.map(toFunc).cache()

   # KmeansRdd=baseRdd.map(KMeansCluster).cache()
  #  KmeansRdd.saveAsTextFile("KMeans")
    
    LDARdd=baseRdd.map(LDA).cache()
    LDARdd.saveAsTextFile("LDA")
    
    to=MainRdd.map(lambda entry:(entry.to_id),1)\
        .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
        .sortBy((lambda (to_id, frequency): frequency), ascending=False)
 
    ebody=MainRdd.map(lambda entry:(entry.body))
    
    user=MainRdd.map(lambda entry:(entry.from_id),1)\
        .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
        .sortBy((lambda (from_id, frequency): frequency), ascending=False)

    word_count=MainRdd.map(lambda entry:(entry.wordcount))
    
    subject=MainRdd.map(lambda entry:(entry.subject))
    
    sentmails = MainRdd\
        .map(lambda entry: (entry.from_id,(1,set([entry.to_id]))))\
        .reduceByKey(lambda (frequency1, to_id_set1), (frequency2, to_id_set2):(frequency1 + frequency2, to_id_set1 | to_id_set2))\
        .sortBy((lambda ((from_id), frequency): from_id), ascending=False)
    
    receivedmails = MainRdd\
            .map(lambda entry: (entry.to_id,(1,set([entry.from_id]))))\
        .reduceByKey(lambda (frequency1, from_id_set1), (frequency2, from_id_set2):(frequency1 + frequency2, from_id_set1 | from_id_set2))\
        .sortBy((lambda ((to_id), frequency): to_id), ascending=False)
    
    received_freq = MainRdd\
        .map(lambda entry: ((entry.from_id, entry.to_id), 1))\
        .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
        .sortBy((lambda ((from_id, to_id), frequency): from_id), ascending=False)
        
    sent_freq = MainRdd\
        .map(lambda entry: ((entry.to_id, entry.from_id), 1))\
        .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
        .sortBy((lambda ((to_id, from_id), frequency): to_id), ascending=False)
    
    toFrequencyByfromid = received_freq\
                .map(lambda ((from_id, to_id), frequency): (from_id, (to_id, frequency)))\
                                    .groupByKey()\
                                    .mapValues(list)
    
    fromFrequencyBytoid = sent_freq\
                .map(lambda ((to_id, from_id), frequency): (to_id, (from_id, frequency)))\
                                    .groupByKey()\
                                    .mapValues(list)
    sortedfromid=MainRdd.map(lambda entry:(entry.from_id,1))\
                                .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
                                .sortBy((lambda (from_id, frequency): frequency), ascending=False)


    joinedfromFrequencies = sortedfromid.join(toFrequencyByfromid)\
                            .sortBy((lambda (from_id, (from_id_frequency, to_id_Frequency)): 
                                from_id_frequency), ascending=False)
    joinedfromFrequencies.saveAsTextFile("mails_sent")                             
    
    sortedtoid=MainRdd.map(lambda entry:(entry.to_id,1))\
                                .reduceByKey(lambda frequency1, frequency2: frequency1 + frequency2)\
                                .sortBy((lambda (to_id, frequency): frequency), ascending=False)
    

    joinedtoFrequencies = sortedtoid.join(fromFrequencyBytoid)\
                            .sortBy((lambda (to_id, (to_id_frequency, from_id_Frequency)): 
                                to_id_frequency), ascending=False)
    joinedtoFrequencies.saveAsTextFile("mails_received")                                 

    numberofmails = MainRdd\
        .map(lambda entry: (entry.filename,(1,set([entry.from_id]))))\
        .reduceByKey(lambda (frequency1, from_id_set1), (frequency2, from_id_set2):(frequency1 + frequency2, from_id_set1 | from_id_set2))\
        .sortBy((lambda ((filename), frequency): filename), ascending=False)

    numberofmails.saveAsTextFile("numberofemails")
    
    averagewordcount = MainRdd\
        .map(lambda entry: (entry.filename,(1,set([entry.wordcount]))))\
        .reduceByKey(lambda (frequency1, wordcount_set1), (frequency2, wordcount_set2):(frequency1 + frequency2, wordcount_set1 | wordcount_set2))\
        .sortBy((lambda ((filename), frequency): filename), ascending=False)

    averagewordcount.saveAsTextFile("wordcount")    
    