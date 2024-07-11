#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
df


# In[6]:


x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)


# In[7]:


df.loc[(df["language"] == "English")]


# In[11]:


user = input("Enter a Sentence: ")
data = cv.transform([user]).toarray()
output = nb.predict(data)
print(output)


# In[ ]:




