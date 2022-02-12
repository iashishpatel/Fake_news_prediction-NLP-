#!/usr/bin/env python
# coding: utf-8

# # Fake News Prediction
# 
# **Problem statement-->**
# The authenticity of Information has become a longstanding issue affecting businesses and society, both for printed and digital media. On social networks, the reach and effects of information spread occur at such a fast pace and so amplified that distorted, inaccurate, or false information acquires a tremendous potential to cause real-world impacts, within minutes, for millions of users. Recently, several public concerns about this problem and some approaches to mitigate the problem were expressed.
# 
# The sensationalism of not-so-accurate eye-catching and intriguing headlines aimed at retaining the attention of audiences to sell information has persisted all throughout the history of all kinds of information broadcast. On social networking websites, the reach and effects of information spread are however significantly amplified and occur at such a fast pace, that distorted, inaccurate, or false information acquires a tremendous potential to cause real impacts, within minutes, for millions of users.
# 
# **Objective-->**
# Our sole objective is to classify the news from the dataset as fake or true news.
# Extensive EDA of news
# Selecting and building a powerful model for classification
# 
# 
# ![jon-tyson-AN7CTlQaRs8-unsplash.jpg](attachment:jon-tyson-AN7CTlQaRs8-unsplash.jpg)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Dataframe 1

# In[2]:


df1 = pd.read_csv("True.csv")
df1.head()


# In[3]:


df1["answer"] = 1
df1.head()


# In[4]:


df1 = df1.drop(["title","subject","date"],axis=1)
df1.head()


# ## Dataframe 2

# In[5]:


df2 = pd.read_csv("Fake.csv")
df2.head()


# In[6]:


df2["answer"] = 0
df2.head()


# In[7]:


df2 = df2.drop(["title","subject","date"],axis=1)
df2.head()


# # Merge two Dataframe together
# 

# In[8]:


df = pd.concat([df1,df2],ignore_index=True)
df.head()


# In[9]:


df.tail()


# ## Word preprocessing 

# In[10]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


# In[11]:


df = df.sample(1000)
df = df.reset_index()


# In[12]:


porter_s = PorterStemmer()
new_df = []
for j in range(len(df["text"])):
        data = re.sub("[^a-zA-Z]"," ",df["text"][j])
        data = data.lower()
        data = data.split()
        data = [porter_s.stem(word)for word in data if word not in set(stopwords.words("english"))]
        data = " ".join(data)
        new_df.append(data)


# In[13]:


new_df[0]


# In[14]:


new_df[8]


# ## Onehot representation 

# In[15]:


get_ipython().system('pip install tensorflow')


# In[16]:


x = new_df
y = df["answer"]


# In[17]:


import tensorflow as tf


# In[18]:


from tensorflow.keras.preprocessing.text import one_hot 


# In[19]:


voc_size = 10000

onehot_repr = [one_hot(word,voc_size)for word in x]
len(onehot_repr[0])


# In[20]:


len(onehot_repr[5])


# In[21]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
dimension = 1000
embedded_docs = pad_sequences(onehot_repr,padding="pre",maxlen=dimension)


# In[22]:


embedded_docs[0:10]


# ## Embedding Representation

# In[23]:


#creating model 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[24]:


embedding_feature = 800
model = Sequential()
model.add(Embedding(voc_size,embedding_feature,input_length = dimension))
model.add(LSTM(1200))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
print(model.summary())


# ## Fitting Training Data And Testing Model Using Testing Data

# In[25]:


import numpy as np
x = np.array(embedded_docs)
y = np.array(y)


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=-0)


# In[27]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)


# In[ ]:




