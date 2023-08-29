#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[3]:


# Load the Excel file into a DataFrame
data = pd.read_excel('NLP_Data.xlsx')
X = data['Description of the Grievance']
y = data['Grievance Category']


# In[4]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Text cleaning
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text


# In[6]:


# Lemmatization
lemmatizer = WordNetLemmatizer()


# In[7]:


def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


# In[8]:


import numpy as np

# Remove rows with NaN values from X_train and X_test
X_train_clean = X_train.dropna()
X_test_clean = X_test.dropna()

# Or, impute NaN values with an empty string
X_train_imputed = X_train.fillna('')
X_test_imputed = X_test.fillna('')


# In[9]:


# Check data types of columns in X_train and X_test
print(X_train.dtypes)
print(X_test.dtypes)


# In[10]:


X_train_unicode = X_train.astype(str)
X_test_unicode = X_test.astype(str)


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


vectorizer = TfidfVectorizer(max_features=5000)


# In[13]:


X_train_tfidf = vectorizer.fit_transform(X_train_clean)


# In[14]:


# Extracting columns
descriptions = data['Description of the Grievance']
categories = data['Grievance Category']


# In[15]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[16]:


# Initialize NLTK and download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

# Text cleaning, tokenization, and lowercasing
cleaned_texts = []
cleaned_categories = []  # To store valid categories corresponding to cleaned texts
for description, category in zip(descriptions, categories):
    if isinstance(description, str) and isinstance(category, str):  # Check if both description and category are strings
        words = word_tokenize(description)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        words = [wnl.lemmatize(word) for word in words]
        cleaned_texts.append(" ".join(words))
        cleaned_categories.append(category)  # Store the corresponding category
    else:
        print("Skipping invalid description or category:", description, category)


# In[17]:


# Remove any remaining NaN or empty string values
cleaned_texts = [text for text in cleaned_texts if text]
cleaned_categories = [category for category in cleaned_categories if isinstance(category, str)]


# In[19]:


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)


# In[20]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, cleaned_categories, test_size=0.2, random_state=42)


# In[21]:


from xgboost import XGBClassifier


# In[22]:


# Convert sparse matrices to dense arrays for imputation
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()


# In[23]:


from sklearn.impute import SimpleImputer


# In[24]:


# Handle missing values in dense arrays
imputer = SimpleImputer(strategy='constant', fill_value=0)  # Replace NaN with 0
X_train_imputed = imputer.fit_transform(X_train_dense)
X_test_imputed = imputer.transform(X_test_dense)


# In[25]:


# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42)


# In[26]:


# Convert boolean labels to integers (1 for True, 0 for False)
y_train_int = [1 if label == 'True' else 0 for label in y_train]
y_test_int = [1 if label == 'True' else 0 for label in y_test]

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42)

# Train the classifier
xgb_classifier.fit(X_train_imputed, y_train_int)  # Use y_train_int instead of y_train

# Predict using the classifier
xgb_predictions = xgb_classifier.predict(X_test_imputed)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test_int, xgb_predictions)  # Use y_test_int instead of y_test

print("XGBoost Accuracy:", xgb_accuracy)


# In[27]:


# Calculate accuracy and convert to percentage
xgb_accuracy = accuracy_score(y_test_int, xgb_predictions) * 100

print("XGBoost Accuracy:", xgb_accuracy, "%")


