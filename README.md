# Classificação de Noticias em Categorias

O objetivo desse trabalho é classificar noticias a partir do título e uma pequena descrição na categoria adequada.

**Hipóteses:**

1. É possivel categorizar notícias based apenas no titulo e uma pequena descrição.

2. Um classificafor treinado no conjunto de dados de noticias é capaz de classificar trechos de noticias na categoria correta.

## Rodar o projeto

Instalar as bibliotecas : `pip install -r requirements.txt`
Run: `streamlit run news_classifier.py`

## Metodologia

### Conjunto de dados

O conjunto de dados utilizado foi o [`News Category Dataset`](https://www.kaggle.com/rmisra/news-category-dataset) disponibilizado na plataforma Kaggle. Esse dataset contém 200k titulos e pequena descrição de noticias coletadas no periodo de 2012 a 2018 obtidos do [`HuffPost`](https://www.huffingtonpost.com/). Cada notícias possui uma categoria entre as 41 categorias disponíveis.

### Preprocessamento

O nosso objetivo é classificar as noticias se baseando no texto, para isso iremos aplicar as seguites etapas de preprocessamento:

- Agrupar titulo e descrição da noticias;
- Transformar texto em minusculo;
- Remoção de pontuação do texto;
- Lematização do texto;
  - Para lematização é preciso fazer a tokenização do texto, porque esse processo ocorre palavra por palavra.

Essas tranformações foram realizadas para padronizar os textos de todas as noticias. A lematização é utilizada para reduzir as palavras para sua forma básica, para ser possivel agrupar palavras parecidas.

### Extração de Características

Para extração de caracteristicas dos texto pre-processados foi utilizado o algoritmo Term Frequency-Inverse Document Frequency (TF-IDF). Essa técnica foi escolhida porque não é muito custosa computacionalmente e porque leva em conta a frequencia com que as palavras aparecem no texto das noticias que é uma vantagem em relação ao Bag-of-Words (BoW).

### Visualização

Visualizações ocorrem com os algoritmos em 2 momentos:

1. Uso da nuvem de palavras após o pre-processamento e remoção de stop-words.
2. Visualização das categorias no plano vetorial de duas domensões após extração de caracteristicas.

Para realização da última visualização é necessário reduzir o espação de caracteriticas para duas dimensões para isso foram utilizadas as seguintes estratégias: `PCA`, `TSE`, `MDS`.

### Classificação

Após a extração de caracteriticas foram utilizados os modelos de aprendizado de máquina classicos para classicar as noticias. Os modelos que foram utilizados foram os seguites.

- `SVM`
- `Random Forest`
- `Naive Bayes`
- `Multilayer Perceptron`

Os modelos foram treinados e em seguida testados, a forma de avaliação utilizada para determinar o modelo com melhor desempenho foi o `f1 score` porque ele é uma média harmonica entre "precision" e "recall". Sendo "precision" a capacidade do modelo de classificar corretamento as labels negativas e "Recall" a capacidade do modelo de classificar corretamente as label positivas.
