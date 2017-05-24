# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:36:50 2016

@author: Yamuna
"""

import os
from stemming.porter2 import stem
from nltk.corpus import stopwords


folder = "all_documents"
num_of_words = 0
stop = stopwords.words('english')

headers = ['Message-ID:','Mime','MIME','Content','X','Date:','Sent:','Cc:','Bcc:','From:','To:','@','X-From:','X-To:','cc:','.xls','download','____']
str_replace = ['-','_','*','[IMAGE]']
mailIDs = ['Message-ID:','From:','Cc:','Bcc:','X-From:','X-To:','cc:','download :']
#fw=open("filewatcher",'a')
parsed_headers = {}
last_key = ""


path = os.path.join(os.path.dirname(__file__), 'maildir')
os.chdir(path)
from pyspark import SparkContext
sc = SparkContext("local", "Parser App")
for ddir in os.listdir(path):
    #if (os.path.isdir(ddir)):
        print("dir-"+ddir)
        for sub_dir in os.listdir(path+"\\"+ddir):
            if sub_dir == folder:
                #print("sub-"+sub_dir)
                for file in os.listdir(path+"\\"+ddir+"\\"+sub_dir):
                    fromset=0
                    subjectset=0
                    if not (os.path.isdir(file)):
                        os.chdir(path+"\\"+ddir+"\\"+sub_dir)
                        fileread=open(file, 'r')
                        data=fileread.readlines()
                        fileread.close()
                        filewrite=open(file,'w')
                        filewrite.close()
                        for line in data:
                            fileappend=open(file,'a')
                            if ('@' in line) and (subjectset==0):
                                if('From:' in line) and (fromset==0) and not ('X-From:' in line):
                                    fromset=1
                                    fromID = line
                                    fileappend.writelines(line)
                                if not any(mail in line for mail in mailIDs):
                                    toID = line.strip().split(',')
                                    toID = [x for x in toID if x]
                                    toIDs =' '.join(word for word in toID)
                                    fileappend.writelines(toIDs)
                            if ("Subject:" in line) and (subjectset==0)and not any(header in line for header in headers):
                                subjectset=1
                                fileappend.writelines("\n"+line)
                            if not any(header in line for header in headers) and not ("Subject:" in line):
                                line=(' '.join([stem(i) for i in line.split() if i not in stop]))
                                for k in str_replace:
                                    if k in line:
                                        line.replace(k,' ')
                                fileappend.writelines(line+" ")
                                fileappend.close()
#fw.close()
