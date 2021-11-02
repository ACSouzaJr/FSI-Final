import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import List
import streamlit as st

# Classification
from classification import classificate_svm, classification_by_classifier, test_hyperparams

# Visualization
from visualization import visualization_by_technique, wordcloud_by_category, plot_all_categories

# Pre processing
from preprocessing import prepare_dataset, preprocessing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    # seleciona apenas texto mais de 12 palavras
    filter_df = dataset[dataset['text_length'] > 12]
    filter_df = filter_df.reset_index(drop=True)

    # Seleciona apenas as 5 maiores categorias
    filter_df = filter_df[filter_df['category'].isin(categories)]
    filter_df.reset_index(drop=True, inplace=True)

    # Seleciona 1000 amostras de cada classe
    filtered_dataset = limit_samples_of_each_category(filter_df, 1000)

    return filtered_dataset


@st.cache(suppress_st_warning=True)
def limit_samples_of_each_category(dataset: pd.DataFrame, quantity: int):
    dataset = dataset.groupby('category').apply(lambda x: x.sample(quantity))
    dataset = dataset.reset_index(drop=True)
    return dataset


def split_dataset(data_text: pd.DataFrame, y: pd.core.series.Series):
    # Split matrix into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        data_text, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def apply_tfidf(x_train: pd.DataFrame, x_test: pd.DataFrame, data_text: pd.DataFrame):
    # Convert data into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')

    # For visualization
    tfidf = vectorizer.fit_transform(data_text)

    # For classification
    vectorizer.fit(x_train)
    tfidf_train = vectorizer.transform(x_train)
    tfidf_test = vectorizer.transform(x_test)
    return tfidf_train, tfidf_test, tfidf, vectorizer

# ----------------------------------------------------------------------------------------------------


DATASET_FILE_PATH = 'https: // github.com/ACSouzaJr/FSI-Final/blob/main/dataset/News_Category_Dataset_v2.json'

# Read Data
st.title("Classificação de Notícias em Categorias")
raw_df = read_data(DATASET_FILE_PATH)

# Apresentação
categories = raw_df['category'].unique()
st.markdown(
    f"O conjunto de dados utilizado foi o [`News Category Dataset`](https://www.kaggle.com/rmisra/news-category-dataset) disponibilizado na plataforma Kaggle. Essa base de dados contém 200k notícias, com o título e uma pequena descrição, coletadas no período de 2012 a 2018 obtidos do [`HuffPost`](https://www.huffingtonpost.com/). Cada notícia pertence a uma categoria dentre as  {len(categories)} categorias disponíveis.")

# Amostra
"Abaixo é possivel ver uma amostra dos dados da base de dados."
st.dataframe(raw_df.head())

"Para classificação das noticias iremos utilizar apenas as colunas `headline` e `short_description`."
raw_df = prepare_dataset(raw_df)

# Categorias
"As categorias existentes na Base da Dados são:"
categories

st.subheader("Análise das categorias")
# Plot de categorias
plot_all_categories(raw_df, color='purple')


"Para reduzir o número de classes da base de dados primeiramente agrupamos os temas parecidos. Reduzindo para 25 categorias"
# Group similar categories
raw_df['category'] = raw_df['category'].replace({
    "HEALTHY LIVING": "WELLNESS",
    "QUEER VOICES": "GROUPS VOICES",
    "BUSINESS": "BUSINESS & FINANCES",
    "PARENTS": "PARENTING",
    "BLACK VOICES": "GROUPS VOICES",
    "THE WORLDPOST": "WORLD NEWS",
    "STYLE": "STYLE & BEAUTY",
    "GREEN": "ENVIRONMENT",
    "TASTE": "FOOD & DRINK",
    "WORLDPOST": "WORLD NEWS",
    "SCIENCE": "SCIENCE & TECH",
    "TECH": "SCIENCE & TECH",
    "MONEY": "BUSINESS & FINANCES",
    "ARTS": "ARTS & CULTURE",
    "COLLEGE": "EDUCATION",
    "LATINO VOICES": "GROUPS VOICES",
    "CULTURE & ARTS": "ARTS & CULTURE",
    "FIFTY": "MISCELLANEOUS",
    "COMEDY": "ENTERTAINMENT",
    "GOOD NEWS": "MISCELLANEOUS"
})

category_changes = pd.DataFrame(
    [["HEALTHY LIVING", "WELLNESS"],
     ["QUEER VOICES", "GROUPS VOICES"],
     ["BUSINESS", "BUSINESS & FINANCES"],
     ["PARENTS", "PARENTING"],
     ["BLACK VOICES", "GROUPS VOICES"],
     ["THE WORLDPOST", "WORLD NEWS"],
     ["STYLE", "STYLE & BEAUTY"],
     ["GREEN", "ENVIRONMENT"],
     ["TASTE", "FOOD & DRINK"],
     ["WORLDPOST", "WORLD NEWS"],
     ["SCIENCE", "SCIENCE & TECH"],
     ["TECH", "SCIENCE & TECH"],
     ["MONEY", "BUSINESS & FINANCES"],
     ["ARTS", "ARTS & CULTURE"],
     ["COLLEGE", "EDUCATION"],
     ["LATINO VOICES", "GROUPS VOICES"],
     ["CULTURE & ARTS", "ARTS & CULTURE"],
     ["FIFTY", "MISCELLANEOUS"],
     ["COMEDY", "ENTERTAINMENT"],
     ["GOOD NEWS", "MISCELLANEOUS"]]
)

category_changes.columns = ["Original", "Nova Categoria"]
category_changes

"Categorias após o agrupamento:"
categories = raw_df['category'].unique()
categories

"Em seguida removemos as noticias em que os textos possuiam menos de 12 palavras"
"25% da base de dados possui text com 12 palavras ou menos."
raw_df['text_length'] = raw_df['text'].apply(lambda x: len(x.split()))
dist_category_df = raw_df.groupby('category').text_length.describe()
st.dataframe(dist_category_df.style.highlight_min(subset=['25%']))

# Exploração dos dados
"Por fim selecionamos 1000 amostras das 6 classes que possuem a maior quantidade de noticias. (Undersample)"
# Category/label selection
categories = ['ENTERTAINMENT', 'GROUPS VOICES',
              'PARENTING', 'POLITICS', 'STYLE & BEAUTY', 'WELLNESS']
filtered_df = select_categories(raw_df, categories)

plot_all_categories(filtered_df, color='purple')

plt.title("Tamanho de texto por categoria")
ax = sns.histplot(data=filtered_df, x="text_length",
                  binwidth=5, hue="category", multiple="stack")
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
st.pyplot(ax.figure)
plt.clf()

# ----------------------------------------------------------------------------------------------------
"Em seguida foi realizado o preprocessamento no texto de cada noticia:"
# Preprocessing
X = preprocessing(filtered_df)
y = filtered_df['category'].values
preprocessed_df = pd.DataFrame({'text': X, 'category': y})

st.dataframe(preprocessed_df.head())

# Data spliting
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Convert to tf-idf
tfidf_train, tfidf_test, tfidf, tfidf_vectorizer = apply_tfidf(
    X_train, X_test, X)

# ----------------------------------------------------------------------------------------------------

st.title("Visualização")

st.subheader("Word Cloud x Category")
category = st.selectbox('Selecione uma categoria:', categories)
wordcloud_by_category(preprocessed_df, category)

st.subheader("Gráficos de dispersão")
techniques_names = ["PCA", "t-SNE", "MDS"]
technique = st.selectbox(
    'Selecione uma técnica de redução de dimensionalidade:', techniques_names)
visualization_by_technique(tfidf, technique, label_names=y)

# ----------------------------------------------------------------------------------------------------

st.title("Classificação")
classifier_names = ["SVM", "Random Forest",
                    "Naive Bayes", "Multilayer Perceptron"]
classifier = st.selectbox(
    'Selecione um classificador:', classifier_names)
st.subheader(classifier)
#classificate_svm(tfidf_train, tfidf_test, y_train, y_test)
model = classification_by_classifier(
    classifier, tfidf_train, tfidf_test, y_train, y_test)


st.title("Descubra a categoria da sua notícia")
news_input = st.text_area("Digite abaixo uma notícia",
                          """ Pete Buttigieg Spends Halloween In Hospital With Son Gus Dressed As Traffic Cone. Chasten Buttigieg wrote that baby Gus "has been having a rough go of it. """)

if len(news_input.strip()) > 12:  # st.button('Classificar') and
    # Preprocessing
    news_input_df = pd.DataFrame({'text': [news_input]})
    preprocessed_news_input = preprocessing(news_input_df)

    # Tf-idf
    tfidf_news_input = tfidf_vectorizer.transform(preprocessed_news_input)

    # Classification
    # Predict labels using test data
    y_pred = model.predict(tfidf_news_input)
    st.write('A notícia escolhida pertence à categoria ', y_pred[0], '.')

# To test the best hyperparams
# test_hyperparams('SVM', tfidf_train, tfidf_test, y_train, y_test)
