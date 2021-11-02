# Classificação de Notícias em Categorias

## Sumário
- Objetivo
- Tecnologias
- Bibliotecas
- Como rodar o projeto
- Especificação dos casos de uso
- Metodologia
- Resultados experimentais
- Deploy
- Referências

## Objetivo

O objetivo desse trabalho é classificar notícias a partir do título e uma pequena descrição na categoria adequada.

**Hipóteses:**

1. É possivel categorizar notícias based apenas no titulo e uma pequena descrição.

2. Um classificafor treinado no conjunto de dados de noticias é capaz de classificar trechos de noticias na categoria correta.

## Tecnologias

Python e Streamlit.

## Bibliotecas

Warnings, Sklearn, TfidfVectorizer, nltk, streamlit, numpy, pandas, scipy, typing, string, seaborn, matplotlib, wordcloud.

## Como rodar o projeto

Instalar as bibliotecas : `pip install -r requirements.txt`
Run: `streamlit run news_classifier.py`

## Especificação dos casos de uso
- Estudo do dataset por meio de classificação e visualização;
- Categorização de notícias de acordo com sua categorias.

## Metodologia

### Conjunto de dados

O conjunto de dados utilizado foi o [`News Category Dataset`](https://www.kaggle.com/rmisra/news-category-dataset) disponibilizado na plataforma Kaggle. Essa base de dados contém 200k notícias, com o título e uma pequena descrição, coletadas no período de 2012 a 2018 obtidos do [`HuffPost`](https://www.huffingtonpost.com/). Cada notícia pertence a uma categoria dentre as 41 categorias disponíveis.

### Preprocessamento

O nosso objetivo é classificar as notícias se baseando no texto. Para isso, iremos aplicar as seguites etapas de pré-processamento:

- Agrupamento do título e da descrição das notícias;
- Transformação do texto para minúsculo;
- Remoção de pontuação do texto;
- Remoção de números do texto;
- Remoção de 'stopwords', de acordo com a biblioteca nltk.
- Lematização do texto;
  - Para lematização é preciso fazer a tokenização do texto, porque esse processo ocorre palavra por palavra.

Essas tranformações foram realizadas para padronizar os textos de todas as notícias. A lematização é utilizada para reduzir as palavras para sua forma básica, para ser possivel agrupar palavras parecidas.

### Extração de Características

Para extração de caracteristicas dos texto pre-processados foi utilizado o algoritmo Term Frequency-Inverse Document Frequency (TF-IDF). Essa técnica foi escolhida porque não é muito custosa computacionalmente e porque leva em conta a frequência com que as palavras aparecem no texto das notícias que é uma vantagem em relação ao Bag-of-Words (BoW).

### Visualização

Visualizações ocorrem com os algoritmos em 2 momentos:

1. Uso da nuvem de palavras após o pre-processamento e remoção de stop-words.
2. Visualização das categorias no plano vetorial de duas dimensões após a extração de características.

Para a realização da última visualização, é necessário reduzir o espaço de características para duas dimensões. Com este objetivo, foram utilizadas as seguintes técnicas: `PCA`, `TSE`, `MDS`.

### Classificação

#### Geral

Após a extração de características, foram utilizados os modelos de aprendizado de máquina clássicos para classificar as notícias. Os modelos que foram utilizados foram os seguintes:

- `SVM`
- `Random Forest`
- `Naive Bayes`
- `Multilayer Perceptron`

Os modelos foram treinados e em seguida testados, a forma de avaliação utilizada para determinar o modelo com melhor desempenho foi o `f1 score` porque ele é uma média harmônica entre "precision" e "recall". Sendo "precision" a capacidade do modelo de classificar corretamento as labels negativas e "Recall" a capacidade do modelo de classificar corretamente as label positivas. Os parâmetros dos classificadores também amplamente testados e escolhidos de acordo com o valor de `f1 score`.

#### Específica

Além da classificação geral, também foi criada a classificação específica, que tem o objetivo de classificar um uma notícia escolhida previamente pelo usuário. Para isso, foi utilizado o classificador `SVM`, o que obteve um dos melhores resultados na etapa anterior.

A notícia escolhida pode ser classificada em uma das 25 categorias. Com o objetivo de diminuir o tempo de processamento, foram utilizadas 1000 instâncias de cada uma das categorias.

## Resultados experimentais

Na classificação, pode-se notar que os classificadores `Random Forest` e `SVM` demonstraram melhores resultados de acordo com a métrica `f1 score`. Na visualização, notou-se que a técnica de redução de dimensionalidade `t-SNE` demonstrou uma melhor disposição de pontos no layout, de acordo com o coeficiente de silhoueta.

## Deploy
https://share.streamlit.io/acsouzajr/fsi-final/main/news_classifier.py

## Referências

https://www.kaggle.com/rmisra/news-category-dataset/code
https://scikit-learn.org/stable/index.html
https://www.nltk.org/
https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1
