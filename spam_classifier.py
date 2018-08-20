
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#import nltk
#nltk.download('stopwords')
#needed to download


# In[3]:


data = pd.read_csv("spam.csv",encoding='latin-1')
#to read csv file using panda
data.head()
#to display first five


# In[4]:


data['v1'].head()
#how to read data of specific column!! head displays first five. if
#you want to display column of name having x data['x]


# In[5]:


data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

#to remove the column we dont need use drop function
#if we want to drop x named column data.drop(["x"],axis=1)
data.head()


# In[6]:


data = data.rename(columns={"v1":"class", "v2":"text"})
#reame column name v1 to class and v2 to text
data.head()


# In[7]:


data['length'] = data['text'].apply(len)
#to create a new column named lenth in given datasets which will be eqal to 
#equal to lenth of messages in corrosponding text rows
data.head()


# In[8]:


def pre_process(text):
    #a preprocessing funtion to remove punctuation and word like
    #is am are using nltk libraries stopwords dictionaries, to stemize 
    # playing to play or played to play using nltk libraries stemmer
    txt1= text
    #text is string passed containing email
    #print (txt1)
   # print (len(txt1)) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    #remove punctuation like , . ?
   # print (text)
   # print (len(text))
    
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    # text.split() split each string i.e message in tokenisation i.e in word
    #  and check that if lowercase of that word is present in stopword
    # dictionaries of nltk python module if prsent then remove it else split and add in list
    
   # print (txt1)
   # print (text)
    #text is now a list of filtered words
    
    words = ""
    #words is empty string to again convert list of words i.e text back to string after stemming
    
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    #words in string made from stemming words which is stored in list text 
    #print (words)
    return words 


# In[9]:


textFeatures = data['text'].copy()
# textFeatures is list of strings containing message
textFeatures = textFeatures.apply(pre_process)

#call preprocess string/messages wise stored in textFeatures


# In[10]:


print( type(textFeatures))


# In[11]:


vectorizer = TfidfVectorizer("english")
print(len(textFeatures))
features = vectorizer.fit_transform(textFeatures)
print( type(features))
#feature is matrix of vector representation of each strings each word 
# present in textFeatures


# In[16]:


features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3, random_state=111)
# this split the data in features and there corrosponding  labels then
# divide data in two list one for training and other for testing
print(type(labels_train))


# In[13]:


#to train naive bays model for prediction
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.35)
mnb.fit(features_train, labels_train)
#testing naive bayes
prediction = mnb.predict(features_test)
accuracy_score(labels_test,prediction)



# In[14]:


#training and testing using SVM classifier
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)
accuracy_score(labels_test,prediction)

