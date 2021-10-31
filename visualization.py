from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from typing import List
import streamlit as st

import pandas as pd
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

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
def visualization_by_technique(tfidf: sp.sparse.csr.csr_matrix, technique: str, label_names):
    if technique == 'PCA':
        generate_pca_visualization(tfidf, label_names)
    elif technique == 'MDS':
        generate_mds_visualization(tfidf, label_names)
    else:
        generate_tsne_visualization(tfidf, label_names)


def generate_tsne_visualization(tfidf: sp.sparse.csr.csr_matrix, label_names):
    tsne_tfidf = TSNE(n_components=2, learning_rate=100,
                      perplexity=40).fit_transform(tfidf)
    st.write('Silhouette Score:', silhouette_score(tsne_tfidf, y))
    data_text = pd.DataFrame(tsne_tfidf)
    data_text['label'] = label_names
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)


def generate_pca_visualization(tfidf: sp.sparse.csr.csr_matrix, label_names):
    not_sparse_tfidf = TruncatedSVD(
        n_components=2, n_iter=7).fit_transform(tfidf)
    pca_tfidf = PCA(n_components=2).fit_transform(not_sparse_tfidf)
    st.write('Silhouette Score:', silhouette_score(pca_tfidf, y))
    data_text = pd.DataFrame(pca_tfidf)
    data_text['label'] = label_names
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)


def generate_mds_visualization(tfidf: sp.sparse.csr.csr_matrix, label_names):
    mds_tfidf = MDS(n_components=2).fit_transform(tfidf.toarray())
    st.write('Silhouette Score:', silhouette_score(mds_tfidf, y))
    data_text = pd.DataFrame(mds_tfidf)
    data_text['label'] = label_names
    data_text.columns = ['x', 'y', 'label']
    plt.figure(figsize=(13, 7))
    fig = sns.scatterplot(data=data_text, x='x', y='y', hue='label')
    st.pyplot(fig.figure)