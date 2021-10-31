import warnings
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from typing import List
import streamlit as st

import numpy as np
import pandas as pd
import scipy as sp
import string

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

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


@ st.cache(suppress_st_warning=True)
def limit_samples_of_each_category(dataset: pd.DataFrame):
    dataset = dataset.groupby('category').apply(lambda x: x.sample(200))
    return dataset


@ st.cache(suppress_st_warning=True)
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

# Plot categories graph


@st.cache(suppress_st_warning=True)
def plot_by_category(dataset: pd.DataFrame, category: List[str]):
    plt.title('Amount of News Based On Category')
    plt.xticks(rotation=90)
    fig = sns.countplot(data=dataset[list(category)], x='category')
    st.pyplot(fig.figure)

# Plot wordcloud


def wordcloud_by_category(dataset: pd.DataFrame, category: List[str]):
    selected_df = dataset[dataset['category'] == category]
    text = " ".join(selected_df['text'].to_numpy())
    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=set(STOPWORDS), max_font_size=100,
                          max_words=100, width=1000, height=328, background_color="white").generate(text)
    # Display the generated image:
    fig, ax = plt.subplots()
    plt.figure(figsize=[20, 5])
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


# Visualization - Scatter plot
def visualization_by_technique(tfidf: sp.sparse.csr.csr_matrix, technique: str):
    if technique == 'PCA':
        generate_pca_visualization(tfidf)
    elif technique == 'MDS':
        generate_mds_visualization(tfidf)
    else:
        generate_tsne_visualization(tfidf)

# @st.cache(suppress_st_warning=True)


def generate_tsne_visualization(tfidf: sp.sparse.csr.csr_matrix):
    tsne_tfidf = TSNE(n_components=2, learning_rate=100,
                      perplexity=40).fit_transform(tfidf)
    st.write('Silhouette Score:', silhouette_score(tsne_tfidf, y))
    data_text = pd.DataFrame(tsne_tfidf)
    data_text['label'] = y
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)

# @st.cache(suppress_st_warning=True)


def generate_pca_visualization(tfidf: sp.sparse.csr.csr_matrix):
    not_sparse_tfidf = TruncatedSVD(
        n_components=2, n_iter=7).fit_transform(tfidf)
    pca_tfidf = PCA(n_components=2).fit_transform(not_sparse_tfidf)
    st.write('Silhouette Score:', silhouette_score(pca_tfidf, y))
    data_text = pd.DataFrame(pca_tfidf)
    data_text['label'] = y
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)

# @st.cache(suppress_st_warning=True)


def generate_mds_visualization(tfidf: sp.sparse.csr.csr_matrix):
    mds_tfidf = MDS(n_components=2).fit_transform(tfidf.toarray())
    st.write('Silhouette Score:', silhouette_score(mds_tfidf, y))
    data_text = pd.DataFrame(mds_tfidf)
    data_text['label'] = y
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)

# Classification


def plot_confusion_matrix(classifier, X_test, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier, X_test, y_test, display_labels=classifier.classes_, cmap=plt.cm.Blues)
    st.pyplot(disp.figure_)


@st.cache(suppress_st_warning=True)
def classificate_svm_news_input(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, label_names, y_train):
    # Define a Support Vector Machine classifier
    svm_clf = SVC(C=1000, gamma=0.001, kernel='sigmoid')
    # Apply SVM
    svm_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_pred = svm_clf.predict(tfidf_test)
    st.write('A notícia escolhida pertence à categoria ', y_pred[0], '.')


def classificate_svm(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, labels_names):
    # Define a Support Vector Machine classifier
    svm_clf = SVC(C=1000, gamma=0.001, kernel='sigmoid')
    # Apply SVM
    svm_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, svm_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    st.write('Classification report: ')
    # print('Classes: ', svm_clf.classes_)
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df
    # Print Confusion Matrix
    plot_confusion_matrix(svm_clf, tfidf_test, y_test)


def classificate_rf(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, labels_names):
    # Define a Random Forest classifier
    rf_clf = RandomForestClassifier(
        criterion='gini', max_depth=None, max_features='log2', n_estimators=300, random_state=0)
    # Apply Random Forest
    rf_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, rf_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '## Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df
    # Print Confusion Matrix
    plot_confusion_matrix(rf_clf, tfidf_test, y_test)


def classificate_nb(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, labels_names):
    mnb_clf = MultinomialNB(alpha=0.1)
    # Apply Multinomial Naive Bayes
    mnb_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, mnb_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '## Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df
    # Print Confusion Matrix
    plot_confusion_matrix(mnb_clf, tfidf_test, y_test)


def classificate_mlp(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, labels_names):
    # Define a MLPClassifier classifier
    mlp_clf = MLPClassifier()
    # Apply SVM
    mlp_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, mlp_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '## Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df
    # Print Confusion Matrix
    plot_confusion_matrix(mlp_clf, tfidf_test, y_test)

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

plt.title('Amount of News Based On Category')
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
category = st.selectbox('Selecione a categoria', label_names)
wordcloud_by_category(preprocessed_df, category)

st.subheader("Gráficos de dispersão")
techniques_names = ["t-SNE", "PCA", "MDS"]
technique = st.selectbox(
    'Selecione a técnica de redução de dimensionalidade', techniques_names)
# visualization_by_technique(tfidf, technique)

st.title("Classificação")
st.subheader("SVM")
classificate_svm(tfidf_train, tfidf_test, y)

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
        tfidf_train_news, tfidf_test_news_input, label_names, y_train_news)
