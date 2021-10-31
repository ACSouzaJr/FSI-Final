import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from typing import List
import streamlit as st

# Classification
from classification import classificate_svm, classificate_svm_news_input, classificate_mlp, classificate_nb, classificate_rf

# Visualization
from visualization import visualization_by_technique, wordcloud_by_category

import pandas as pd
import string

import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('wordnet')

warnings.filterwarnings("ignore")

# Read data


@st.cache(suppress_st_warning=True)
def read_data(file_path: str):
    return pd.read_json(file_path, lines=True)

# Preprocessing


def select_categories(dataset: pd.DataFrame, categories: List[str]):
    filtered_dataset = dataset[dataset['category'].isin(categories)]
    filtered_dataset.reset_index(drop=True, inplace=True)
    return filtered_dataset


@st.cache(suppress_st_warning=True)
def limit_samples_of_each_category(dataset: pd.DataFrame):
    dataset = dataset.groupby('category').apply(lambda x: x.sample(200))
    return dataset


@st.cache(suppress_st_warning=True)
def preprocessing(dataset: pd.DataFrame, entire_dataset: bool):
    if entire_dataset:
        # Drop unused columns
        df = dataset.drop(columns=['link', 'date', 'authors'], errors='ignore')
        # Join headline and short description
        text_data = df.apply(lambda row: row.headline +
                             '. ' + row.short_description, axis=1)
    else:
        # Convert to lower case
        text_data = dataset[0]

    # Convert to lower case
    text_data = [text.lower() for text in text_data]
    # Strip all punctuation
    table = str.maketrans('', '', string.punctuation)
    text_data = [text.translate(table) for text in text_data]
    # Convert text to tokens
    text_data = [nltk.word_tokenize(text) for text in text_data]
    # Lemmatizing
    text_data = [[WordNetLemmatizer().lemmatize(word, 'n')
                  for word in text] for text in text_data]
    # Convert tokens to sentences
    text_data = [TreebankWordDetokenizer().detokenize(text)
                 for text in text_data]
    return text_data


def split_dataset(data_text: pd.DataFrame, y: pd.core.series.Series):
    # Split matrix into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        data_text, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def apply_tfidf(x_train: pd.DataFrame, x_test: pd.DataFrame, data_text: pd.DataFrame):
    # Convert data into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    # For classification
    vectorizer.fit(x_train)
    tfidf_train = vectorizer.transform(x_train)
    tfidf_test = vectorizer.transform(x_test)
    # For visualization
    tfidf = vectorizer.fit_transform(data_text)
    return tfidf_train, tfidf_test, tfidf

# ----------------------------------------------------------------------------------------------------


DATASET_FILE_PATH = 'dataset/News_Category_Dataset_v2.json'

# Read Data
st.title("Informações do dataset")
raw_df = read_data(DATASET_FILE_PATH)
categories = raw_df['category'].unique()
st.write(f"Número de classes totais no dataset: {len(categories)}")
categories

# Category/label selection
label_names = ['WORLD NEWS', 'ARTS & CULTURE', 'SCIENCE', 'TECH']
filtered_df = select_categories(raw_df, label_names)

st.write("Primeiros 5 registros:")
st.write(filtered_df.head())
st.write(
    "Para classificação iremos utilizar apenas as colunas ['headline', 'short_description']")

plt.title('Quantidade de notícias por categoria')
plt.xticks(rotation=90)
fig = sns.countplot(data=raw_df, x='category', color='purple')
st.pyplot(fig.figure)

# Preprocessing
X = preprocessing(filtered_df, True)
y = filtered_df['category'].to_numpy()
preprocessed_df = pd.DataFrame({'text': X, 'category': y})

st.write(preprocessed_df.head())

# Data spliting
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Convert to tf-idf
tfidf_train, tfidf_test, tfidf = apply_tfidf(X_train, X_test, X)

st.title("Visualização")

st.subheader("Word Cloud x Category")
category = st.selectbox('Selecione uma categoria:', label_names)
wordcloud_by_category(preprocessed_df, category)

st.subheader("Gráficos de dispersão")
techniques_names = ["t-SNE", "PCA", "MDS"]
technique = st.selectbox(
    'Selecione uma técnica de redução de dimensionalidade:', techniques_names)
# visualization_by_technique(tfidf, technique)

st.title("Classificação")
classifier_names = ["SVM", "Random Forest",
                    "Naive Bayes", "Multilayer Perceptron"]
classifier = st.selectbox(
    'Selecione um classificador:', classifier_names)
st.subheader(classifier)
classificate_svm(tfidf_train, tfidf_test, y_train, y_test)
# classification_by_classifier(classifier, tfidf_train, tfidf_test, y)

st.title("Descubra a categoria da sua notícia")
news_input = st.text_area("Digite abaixo uma notícia", '')
news_input = pd.DataFrame({news_input})

if st.button('Classificar'):
    # Preprocessing
    limited_df = limit_samples_of_each_category(raw_df)
    X_news = preprocessing(limited_df, True)
    y_news = limited_df['category'].to_numpy()
    preprocessed_df = pd.DataFrame({'text': X_news, 'category': y_news})
    preprocessed_news_input = preprocessing(news_input, False)

    # Data spliting
    X_train_news, X_test_news, y_train_news, y_test_news = split_dataset(
        X_news, y_news)

    # Tf-idf
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(X_train_news)
    tfidf_train_news = vectorizer.transform(X_train_news)
    tfidf_test_news_input = vectorizer.transform(preprocessed_news_input)

    # Classification
    classificate_svm_news_input(
        tfidf_train_news, tfidf_test_news_input, y_train_news)
