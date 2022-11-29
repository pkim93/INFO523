#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Download the dataset from https://www.yelp.com/dataset.
#It will be a json file. The purpose of this is to convert json to csv file for later.

import pandas as pd

business_json_path = 'D:\Pureum\Documents\yelp_database\yelp_academic_dataset_business.json'
df_b = pd.read_json(business_json_path, lines=True)
df_b


# In[2]:


#Let's only include the restaurants that are currently open.
# 1 = open, 0 = closed
df_b = df_b[df_b['is_open']==1]


# In[3]:


#Getting rid of unnecessary columns
drop_columns = ['hours','is_open','review_count']
df_b = df_b.drop(drop_columns, axis=1)


# In[4]:


#This is to only pull the reviews from restaurants.

df_explode = df_b.assign(categories = df_b.categories.str.split(', ')).explode('categories')

df_explode.categories.value_counts()

df_explode[df_explode.categories.str.contains('Restaurant', case=True,na=False)].categories.value_counts()

business_Restaurants = df_b[df_b['categories'].str.contains('Restaurants', case=False, na=False)]

review_json_path = 'D:\Pureum\Documents\yelp_database\yelp_academic_dataset_review.json'


# In[5]:


size = 1000000 #Let's pull about 1 million reviews (My computer crashes if it's bigger)
review = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)


# In[6]:


# There are multiple chunks to be read
chunk_list = []
for chunk_review in review:
    # Renaming column name to avoid conflict with business overall star rating
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(business_Restaurants, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)
    
# After trimming down the review file, concatenate all relevant data back to one dataframe
df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)


# In[8]:


df.head()


# In[9]:


#Converting this to CSV because I like working with CSV files.
#But the CSV file is about 6.9GB in size.

csv_name = "yelp_reviews_Restaurants_categories_INFO523.csv"
df.to_csv(csv_name, index=False)


# In[10]:


#Load the necessary packages

import pandas as pd
import random
import csv


# In[11]:


yelp_csv_path = 'D:\Pureum\Documents\yelp_database\yelp_reviews_Restaurants_categories_INFO523.csv'


# In[12]:


#Read the CSV file and put it in a variable
yelp_dataset = pd.read_csv(yelp_csv_path)


# In[18]:


#Randomly select 10,000 reviews from the original dataset for faster processing
yelp = yelp_dataset.sample(10000, random_state=1)
yelp.head()


# In[15]:


#Prep for sentiment analysis using multinomial naive Bayes model.
#Please ignore the redudant loading. I just like to have them to remind myself.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


# In[19]:


#Check how many rows and columns are in the dataframe
yelp.shape


# In[21]:


#Find out the overall text length of the reviews
yelp['text_length'] = yelp['text'].apply(len)
yelp.head()


# In[22]:


#Explore the dataset
g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text_length', bins=50)


# In[23]:


#Boxplot for each of the ratings
sns.boxplot(x='stars', y='text length', data=yelp)


# In[28]:


#Grab reviews that are either 1 or 5 stars from the yelp dataframe

yelp_class = yelp[(yelp['review_stars'] == 1) | (yelp['review_stars'] == 5)]
yelp_class.shape


# In[29]:


#Create x and y for the classification task
X = yelp_class['text']
y = yelp_class['review_stars']


# In[37]:


#To start preparing for classification task, the text has to be converted into vectors.
#To do so we are going to take the bag of words approach.
#The following function will remove all punctuation marks, stop words, and return clean list of words.

import string
def text_process(text):

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[39]:


#We have our reviews as lists of token. To work the algorithm on our text, we need to convert the reviews into vector.
#We will use CountVectorizer for that.

from sklearn.feature_extraction.text import CountVectorizer
review_transformer = CountVectorizer(analyzer=text_process).fit(X)


# In[42]:


# Transform X dataframe into a sparse matrix
X = review_transformer.transform(X)


# In[43]:


#split X and y into a training and a test set. 30% of the dataset will be used for testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[44]:


#Use multinomial naive Bayes model and fit it into our training data
#And train the model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[45]:


#Check how well the model predicts
predictions = nb.predict(X_test)


# In[48]:


#Evaluate our predictions against the actual ratings using confusion_matrix and classification_report
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

#This means that our model can predict whether a user liked a local business or not, based on the review.


# In[ ]:




