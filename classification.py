from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import streamlit as st
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def classification_by_classifier(classifier: str, tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_test):
    if classifier == 'Random Forest':
        classificate_rf(tfidf_train, tfidf_test, y_test)
    elif classifier == 'Naive Bayes':
        classificate_nb(tfidf_train, tfidf_test, y_test)
    elif classifier == 'Multilayer Perceptron':
        classificate_mlp(tfidf_train, tfidf_test, y_test)
    else:
        classificate_svm(tfidf_train, tfidf_test, y_test)


def plot_confusion_matrix(classifier, X_test, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier, X_test, y_test, display_labels=classifier.classes_, cmap=plt.cm.Blues)
    st.pyplot(disp.figure_)


@st.cache(suppress_st_warning=True)
def classificate_svm_news_input(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train):
    # Define a Support Vector Machine classifier
    svm_clf = SVC(C=1000, gamma=0.001, kernel='sigmoid')
    # Apply SVM
    svm_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_pred = svm_clf.predict(tfidf_test)
    st.write('A notícia escolhida pertence à categoria ', y_pred[0], '.')


def classificate_svm(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
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


def classificate_rf(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
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


def classificate_nb(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
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


def classificate_mlp(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
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