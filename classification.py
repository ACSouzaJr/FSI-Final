from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import streamlit as st
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


def classification_by_classifier(classifier: str, tfidf_train, tfidf_test, y_train, y_test):
    if classifier == 'Random Forest':
        return classificate_rf(tfidf_train, tfidf_test, y_train, y_test)
    elif classifier == 'Naive Bayes':
        return classificate_nb(tfidf_train, tfidf_test, y_train, y_test)
    elif classifier == 'Multilayer Perceptron':
        return classificate_mlp(tfidf_train, tfidf_test, y_train, y_test)
    else:
        return classificate_svm(tfidf_train, tfidf_test, y_train, y_test)


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
    '### Classification report: '
    # print('Classes: ', svm_clf.classes_)
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)
    # Print Confusion Matrix
    plot_confusion_matrix(svm_clf, tfidf_test, y_test)
    return svm_clf


def classificate_rf(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
    # Define a Random Forest classifier
    rf_clf = RandomForestClassifier(
        criterion='gini', max_depth=None, max_features='log2', n_estimators=300, random_state=0)
    # Apply Random Forest
    rf_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, rf_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '### Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)
    # Print Confusion Matrix
    plot_confusion_matrix(rf_clf, tfidf_test, y_test)
    return rf_clf


def classificate_nb(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
    mnb_clf = MultinomialNB(alpha=0.1)
    # Apply Multinomial Naive Bayes
    mnb_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, mnb_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '### Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)
    # Print Confusion Matrix
    plot_confusion_matrix(mnb_clf, tfidf_test, y_test)
    return mnb_clf


def classificate_mlp(tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
    # Define a MLPClassifier classifier
    mlp_clf = MLPClassifier()
    # Apply SVM
    mlp_clf.fit(tfidf_train, y_train)
    # Predict labels using test data
    y_true, y_pred = y_test, mlp_clf.predict(tfidf_test)
    # Print a text report showing the main classification metrics
    '### Classification report: '
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)
    # Print Confusion Matrix
    plot_confusion_matrix(mlp_clf, tfidf_test, y_test)
    return mlp_clf

# --------------------------------------------------------------------------------


def test_hyperparams(classifier: str, tfidf_train: sp.sparse.csr.csr_matrix, tfidf_test: sp.sparse.csr.csr_matrix, y_train, y_test):
    tuned_parameters = {}
    classification = None

    if classifier == 'Random Forest':
        tuned_parameters = {'n_estimators': [100, 200, 300, 400, 500], "max_depth": [
            3, 4, 5, 6, 7, 8, None], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}
        classification = RandomForestClassifier()
    elif classifier == 'Naive Bayes':
        tuned_parameters = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
        classification = MultinomialNB()
    elif classifier == 'Multilayer Perceptron':
        tuned_parameters = {'learning_rate': [
            'constant', 'invscaling', 'adaptive'], 'alpha': [0.001, 0.0001, 0.00001, 0.00001]}
        classification = MLPClassifier()
    elif classifier == 'SVM':
        tuned_parameters = {'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], "gamma": [
            1e-3, 1e-4], "C": [1, 10, 100, 1000]}
        classification = SVC()

    score = 'f1'

    st.write("# Tuning hyper-parameters for %s" % score)
    st.write()

    clf = GridSearchCV(classification, tuned_parameters,
                       scoring="%s_macro" % score, n_jobs=-1)
    clf.fit(tfidf_train, y_train)

    st.write("Best parameters set found on development set:")
    st.write()
    st.write(clf.best_params_)
    st.write()
    st.write("Grid scores on development set:")
    st.write()
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        st.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    st.write()

    st.write("Detailed classification report:")
    st.write()
    st.write("The model is trained on the full development set.")
    st.write("The scores are computed on the full evaluation set.")
    st.write()
    y_true, y_pred = y_test, clf.predict(tfidf_test)
    st.write(classification_report(y_true, y_pred))
    st.write()
