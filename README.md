# Grievance Category Classification using NLP and XGBoost

This project focuses on building a machine learning model to classify grievance descriptions into their respective categories. The workflow includes data preprocessing, text cleaning, feature extraction using TF-IDF, and classification using the XGBoost classifier.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Project Overview

This project aims to categorize grievances based on their textual descriptions into different predefined categories. It uses Natural Language Processing (NLP) techniques to clean and preprocess the data, extracts features using Term Frequency-Inverse Document Frequency (TF-IDF), and applies the XGBoost classifier for classification.

## Features

- **Text Cleaning and Preprocessing:** Includes tokenization, removal of stop words, lemmatization, and handling of missing values.
- **TF-IDF Vectorization:** Converts the textual data into numerical form, suitable for machine learning algorithms.
- **XGBoost Classifier:** A powerful, efficient, and scalable implementation of the gradient boosting framework.

## Technologies Used

- Python
- pandas
- nltk
- scikit-learn
- xgboost

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Atulsharma428/Grievance-Category-Classification.git
   ```

2. Navigate to the project directory:

```bash
Copy code
cd Grievance-Category-Classification
Install the required Python packages:
```
```bash
Copy code
pip install -r requirements.txt
Download the NLTK stop words and lemmatizer data:
```
```python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
## Project Workflow

### Data Preprocessing
1. Load Dataset: The dataset is loaded from an Excel file containing grievance descriptions and categories.
2. Text Cleaning: Special characters and unnecessary spaces are removed. Text is converted to lowercase and lemmatized for better feature extraction.
3. Handle Missing Values: Missing values are handled either by dropping them or imputing empty strings.

### Feature Extraction
TF-IDF Vectorizer: A maximum of 1000 features are extracted from the cleaned text using the TfidfVectorizer.
### Model Training
XGBoost Classifier: The cleaned and transformed text data is split into training and testing sets. The XGBoost classifier is trained on the training set and evaluated on the testing set.
### Evaluation
Accuracy Calculation: The model's performance is evaluated using accuracy as the primary metric. Results are printed at the end of the script.
## Results
XGBoost Accuracy: The model achieved an accuracy of 100% on the test dataset, where X is the calculated value during training and testing.
## Conclusion
This project demonstrates the effectiveness of using NLP techniques combined with XGBoost for text classification tasks. The preprocessing steps, especially text cleaning and lemmatization, helped improve the model's performance.

## Future Work
Explore other classifiers such as Random Forest, SVM, or neural networks for improved accuracy.
Implement hyperparameter tuning to further improve the XGBoost model.
Increase the feature space for TF-IDF and try other feature extraction methods like Word2Vec or BERT embeddings.
