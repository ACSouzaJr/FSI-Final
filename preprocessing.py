from typing import List
import pandas as pd
import string
import contractions
import streamlit as st

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from word2number import w2n
from nltk.corpus import stopwords

def expand_contractions(corpus: List[str]):
  return [contractions.fix(text) for text in corpus]

def convert_word_to_number(corpus: List[str]):
  def convert_word_to_number_from_text(text: str):
    new_sentence = []
    for token in text:
      try:
        new_token = str(w2n.word_to_num(token))
      except:
        new_token = token
      new_sentence.append(new_token)
    
    return new_sentence

  return [convert_word_to_number_from_text(text) for text in corpus]

def remove_numbers(corpus: List[str]):
  def remove_numbers_from_text(text: str):
    return [token for token in text if token.isalpha()]

  return [remove_numbers_from_text(text) for text in corpus]

def convert_lower_case(corpus: List[str]):
  return [text.lower() for text in corpus]

def remove_punctuation(corpus: List[str]):
  # Remove artifacts
  new_corpus = [text.replace("'s", "") for text in corpus]
  new_corpus = [text.replace("’s", "") for text in new_corpus]

  table = str.maketrans('', '', string.punctuation)
  return [text.translate(table) for text in new_corpus]

def tokenize(corpus: List[str]):
  return [nltk.word_tokenize(text) for text in corpus]

def lemmatize(corpus: List[str]):
  wordnet_lemmatizer = WordNetLemmatizer()
  return [[wordnet_lemmatizer.lemmatize(word)
                  for word in text] for text in corpus]

def remove_stop_words(corpus: List[str]):
  stop_words = set(stopwords.words('english'))
  return [[word for word in text if not word in stop_words] for text in corpus]

def prepare_dataset(dataset):
  # Drop unused columns
  df = dataset.drop(columns=['link', 'date', 'authors'], errors='ignore')
  # Join headline and short description
  df['text'] = df.apply(lambda row: row.headline +
                        '. ' + row.short_description, axis=1)
  return df

@st.cache(suppress_st_warning=True)
def preprocessing(dataset: pd.DataFrame, entire_dataset: bool):
    
  text_data = dataset['text'].values
  text_data = convert_lower_case(text_data)
  #text_data = expand_contractions(text_data)
  text_data = remove_punctuation(text_data)
  text_data = tokenize(text_data)
  text_data = convert_word_to_number(text_data)
  text_data = remove_numbers(text_data)
  text_data = remove_stop_words(text_data)
  text_data = lemmatize(text_data)

  # Convert tokens to sentences
  text_data = [TreebankWordDetokenizer().detokenize(text)
                for text in text_data]
  return text_data