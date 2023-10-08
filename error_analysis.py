import sys
import pandas as pd
import numpy as np
import models

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Encoder and Vectorizer are used globally(in different functions), so defining here
le = LabelEncoder()
vectorizer = CountVectorizer(binary=True, strip_accents="unicode", max_features=90000)


def train_logistic_regression(X_train_vec, y_train):
    # Train Logistic Regression
    # 1000 because with default (100) MAX_ITER warning was reached
    log_regression = LogisticRegression(max_iter=1000)
    log_regression.fit(X_train_vec, y_train)

    return log_regression


def train_decisionTree(X_train_vec, y_train):
    # Train Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=18)
    decision_tree = decision_tree.fit(X_train_vec, y_train)

    return decision_tree


def train_knn(X_train_vec, y_train):
    # Train K-Nearest Neighbors
    knn = KNeighborsClassifier(
        n_neighbors=147
    )  # Choose K according rule: sqrt(N) where N is number of instances
    knn.fit(X_train_vec, y_train)

    return knn

    
def main():
    df = models.load_file()
    
    df.label = df.label.str.lower()
    df.utterance = df.utterance.str.lower()

    # Optional for comparison
    # df = df.drop_duplicates()
    
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, le = models.preprocess(df)

    y_baseline_1 = models.baseline_model_1(X_test)
    models.assess_performance(y_test, y_baseline_1, "Baseline 1")

    y_baseline_2 = models.baseline_model_2(X_test)
    models.assess_performance(y_test, y_baseline_2, "Baseline 2")

    log_reg = train_logistic_regression(X_train_vec, y_train)
    y_log_reg = log_reg.predict(X_test_vec)
    models.assess_performance(y_test, y_log_reg, "Logistic Regression")

    decision_tree = train_decisionTree(X_train_vec, y_train)
    y_decision_tree = decision_tree.predict(X_test_vec)
    models.assess_performance(y_test, y_decision_tree, "Decision Tree")

    knn = train_knn(X_train_vec, y_train)
    y_knn = knn.predict(X_test_vec)
    models.assess_performance(y_test, y_knn, "K-Nearest Neighbors")
    
    y_test_label = le.inverse_transform(y_test)
    y_baseline_1_label = le.inverse_transform(y_baseline_1)
    y_baseline_2_label = le.inverse_transform(y_baseline_2)
    y_log_reg_label = le.inverse_transform(y_log_reg)
    y_decision_tree_label = le.inverse_transform(y_decision_tree)
    y_knn_label = le.inverse_transform(y_knn)
    
    # Summarize the results from different models for analysis
    df_results = pd.DataFrame({
        'utterance': X_test,
        'true_labels': y_test_label,
        'baseline_1': y_baseline_1_label,
        'baseline_2': y_baseline_2_label,
        'logistic_regression': y_log_reg_label,
        'decision_tree': y_decision_tree_label,
        'knn': y_knn_label
    })
    print(df_results)
    
    
    # Count the frequency of misclassified categories for each model
    for model in ['baseline_2', 'logistic_regression', 'decision_tree', 'knn']:
        hard_to_predict = df_results[df_results[model] != df_results['true_labels']]['true_labels'].value_counts()
        print(f"Hard to predict categories for {model}: \n{hard_to_predict}\n")
        hard_to_predict.to_csv(f'hard_to_predict_dialogs_{model}.csv')

    # Get all misclassified utterances
    all_models = ['logistic_regression', 'decision_tree', 'knn']
    df_results['all_wrong'] = df_results.apply(lambda row: all(row[model] != row['true_labels'] for model in all_models), axis=1)
    misclassified_samples = df_results[df_results['all_wrong']]
    print(f"Samples misclassified by all models: \n{misclassified_samples[['utterance','true_labels']]}\n")
    print(f"Number of samples misclassified by all models: {len(misclassified_samples[['utterance','true_labels']])}")

if __name__ == '__main__':
    main()
    