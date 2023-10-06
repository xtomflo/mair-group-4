import sys
import pandas as pd
import numpy as np

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


def load_file(file_path: str = "dialog_acts.dat") -> pd.DataFrame:
    #
    # Initialize empty lists to store labels and utterances
    labels = []
    utterances = []

    # Read the file line by line and pick labels & utterances into separate lists
    with open(file_path, "r") as f:
        for line in f:
            # Split each line into label and utterance
            parts = line.strip().split(" ", 1)  # Split by first " " - space
            label, utterance = parts
            labels.append(label)
            utterances.append(utterance)

    # Create a Pandas DataFrame from the lists
    df = pd.DataFrame({"label": labels, "utterance": utterances})

    return df


# Converts all the columns in the DataFrame to lowercase
def convert_to_lowercase(df: pd.DataFrame):
    df.label = df.label.str.lower()
    df.utterance = df.utterance.str.lower()

    return df


def split_dataset(df: pd.DataFrame):
    # Split into 85% training & 15% test data
    X_train, X_test, y_train, y_test = train_test_split(
        df["utterance"], df["label"], test_size=0.15, random_state=42
    )

    return (X_train, X_test, y_train, y_test)


def preprocess(df: pd.DataFrame):
    # Preprocessing:
    # 1. Encode Labels
    # 2. Split into test & training sets
    # 3. Embed Sentences to Vectors

    # Fit the encoder with a list of all labels
    le.fit(df["label"].unique())
    # Replace the labels with encodings
    df = df.copy()  # To mute warnings
    df["label"] = le.transform(df["label"])

    # Fit the vectorizer with utterances
    vectorizer.fit(df["utterance"])

    X_train, X_test, y_train, y_test = split_dataset(df)

    # Vectorized train & test utterances
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Baseline models need non-vectorized utterences, ML models need vectors so returning both types
    return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec


def baseline_model_1(X_test):
    # Baseline model No. 1
    # Assume all utterances are "inform" - the most common label.

    # Create an array with length of X_test (all test cases) and fill with encoded version of label "inform"
    inform_encoding = le.transform(["inform"])[0]
    y_predicted = np.full(len(X_test), inform_encoding)

    return y_predicted


def baseline_model_2(X_test):
    # TODO
    # Baseline model 2
    # ack	- "okay" "aha"
    # affirm	- "yes"
    # bye	- "bye", "goodbye"
    # confirm	 - "is it" "was it"
    # deny	- "no" "dont" "don't" "won't"
    # hello	 - "hello" "hi"
    # inform	- "price" "north" "east" "south" "west" "chinese" "type" "mexican" "thai" "cheap" "expensive"
    # negate - "no"
    # null	- "cough"
    # repeat	- "repeat"
    # reqalts	- "how"
    # reqmore	request more suggestions	more
    # request	ask for information	what is the post code
    # restart	attempt to restart the dialog	okay start over
    # thankyou	express thanks	thank you good bye

    # Let's not overthink it, it's just the baseline - dummy is fine.
    # DEF List of ACK keywords
    # DEF List of Affirm keywords
    # .....
    #
    # Go through the X_test row by row.
    # IF keyword for ack exists in the sentence, apply the ack label.
    # ELSE IF keyword for affirm exists in the sentence, apply affirm label
    keyWords = defaultdict(set)
    dict = defaultdict(set)
    keyWords["ack"] = ["okay", "aha"]
    keyWords["affirm"] = ["yes"]
    keyWords["bye"] = ["bye", "goodbye"]
    keyWords["confirm"] = ["is it", "was it"]
    keyWords["deny"] = ["no", "dont", "don't", "won't"]
    keyWords["hello"] = ["hello", "hi"]
    keyWords["inform"] = [
        "price",
        "north",
        "east",
        "south",
        "west",
        "chinese",
        "type",
        "mexican",
        "thai",
        "cheap",
        "expensive",
    ]
    keyWords["negate"] = ["no"]
    keyWords["null"] = ["cough"]
    keyWords["repeat"] = ["repeat", "again"]
    keyWords["reqalts"] = ["how"]
    keyWords["reqmore"] = ["more"]
    keyWords["request"] = ["postcode", "address"]
    keyWords["restart"] = ["restart"]
    keyWords["thankyou"] = ["thanks"]

    for k in keyWords.keys():
        for v in keyWords[k]:
            if v not in dict[le.transform([k])[0]]:
                dict[le.transform([k])[0]].add(v)
    inform_encoding = le.transform(["inform"])[0]
    y_predicted = np.full(len(X_test), -1)
    cnt = 0
    for sent in X_test:
        label = None
        words = sent.split(" ")
        for w in words:
            for k in keyWords.keys():
                if w in keyWords[k]:
                    label = k
                    break
        if label is None:
            y_predicted[cnt] = inform_encoding
        else:
            encoding = le.transform([label])[0]
            y_predicted[cnt] = encoding
        cnt += 1

    return y_predicted


def train_logistic_regression(X_train_vec, y_train):
    # 1000 because with default (100) MAX_ITER warning was reached
    log_regression = LogisticRegression(max_iter=1000)
    log_regression.fit(X_train_vec, y_train)

    return log_regression


def train_decisionTree(X_train_vec, y_train):
    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=18)
    decision_tree = decision_tree.fit(X_train_vec, y_train)

    return decision_tree


def train_knn(X_train_vec, y_train):
    knn = KNeighborsClassifier(
        n_neighbors=147
    )  # Choose K according rule: sqrt(N) where N is number of instances
    knn.fit(X_train_vec, y_train)

    return knn


def assess_performance(y_test, y_predicted, model_name):
    print(f"Results for Model -> {model_name}")

    # Print different scores up to 2 decimal points (.2f)
    precision = precision_score(y_test, y_predicted, average="macro", zero_division=1)
    recall = recall_score(y_test, y_predicted, average="macro")
    f1_score = 2 * precision * recall / (precision + recall)

    print(f" Accuracy Score: {accuracy_score(y_test, y_predicted):.2f}")
    print(f" Precision Score: {precision:.2f}")
    print(f" Recall Score: {recall:.2f}")
    print(f" F1 Score: {f1_score:.2f}")


def predictions_process(df: pd.DataFrame):
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec = preprocess(df)

    y_baseline_1 = baseline_model_1(X_test)
    assess_performance(y_test, y_baseline_1, "Baseline 1")

    y_baseline_2 = baseline_model_2(X_test)
    assess_performance(y_test, y_baseline_2, "Baseline 2")

    log_reg = train_logistic_regression(X_train_vec, y_train)
    y_log_reg = log_reg.predict(X_test_vec)
    assess_performance(y_test, y_log_reg, "Logistic Regression")

    decision_tree = train_decisionTree(X_train_vec, y_train)
    y_decision_tree = decision_tree.predict(X_test_vec)
    assess_performance(y_test, y_decision_tree, "Decision Tree")

    knn = train_knn(X_train_vec, y_train)
    y_knn = knn.predict(X_test_vec)
    assess_performance(y_test, y_knn, "K-Nearest Neighbors")

    return log_reg

    
def main():
    df = load_file()
    df = convert_to_lowercase(df)

    # Optional for comparison
    # df = df.drop_duplicates()
    
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec = preprocess(df)

    y_baseline_1 = baseline_model_1(X_test)
    assess_performance(y_test, y_baseline_1, "Baseline 1")

    y_baseline_2 = baseline_model_2(X_test)
    assess_performance(y_test, y_baseline_2, "Baseline 2")

    log_reg = train_logistic_regression(X_train_vec, y_train)
    y_log_reg = log_reg.predict(X_test_vec)
    assess_performance(y_test, y_log_reg, "Logistic Regression")

    decision_tree = train_decisionTree(X_train_vec, y_train)
    y_decision_tree = decision_tree.predict(X_test_vec)
    assess_performance(y_test, y_decision_tree, "Decision Tree")

    knn = train_knn(X_train_vec, y_train)
    y_knn = knn.predict(X_test_vec)
    assess_performance(y_test, y_knn, "K-Nearest Neighbors")
    
    
    y_test_label = le.inverse_transform(y_test)
    y_baseline_1_label = le.inverse_transform(y_baseline_1)
    y_baseline_2_label = le.inverse_transform(y_baseline_2)
    y_log_reg_label = le.inversve_transform(y_log_reg)
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



if __name__ == '__main__':
    main()