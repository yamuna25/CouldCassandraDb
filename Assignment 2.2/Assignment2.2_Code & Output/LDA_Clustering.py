from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
import os
path = os.path.join(os.path.dirname(__file__), 'maildir')
os.chdir(path)
#vectorizer = TfidfVectorizer(stop_words='english')
headers = ['Message-ID:','Mime','MIME','Content','X','Date:','Sent:','Cc:','Bcc:','From:','To:','@','X-From:','X-To:','cc:','.xls','download','____']
folder = "all_documents"
subject=[]
mail=[]
parsed_mail = {}
from pyspark import SparkContext
sc = SparkContext("local", "Parser App")
for ddir in os.listdir(path):
    #if (os.path.isdir(ddir)):
        print("User: "+ddir)
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
    
# list for tokenized documents in loop
texts = []

# loop through document list
for i in mail:
    
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
print(corpus[0])
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=3))
lda=ldamodel.print_topics(num_topics=3, num_words=3)



