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
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Encoder and Vectorizer are used globally(in different functions), so defining here
le = LabelEncoder()
vectorizer = CountVectorizer(binary=True, strip_accents="unicode", max_features=90000)
log_reg = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=174)

LOG_REG_TRAINED = False
KNN_TRAINED = False


def load_file(file_path: str = "dialog_acts.dat") -> pd.DataFrame:
    # Load the file with labels in utterances

    # Initialize empty lists to store labels and utterances
    labels = []
    utterances = []

    # Read the file line by line and pick labels & utterances into separate lists
    with open(file_path, "r") as f:
        for line in f:
            # Split each line into label and utterance
            parts = line.lower().strip().split(" ", 1)  # Split by first " " - space
            label, utterance = parts
            labels.append(label)
            utterances.append(utterance)

    # Create a Pandas DataFrame from the lists
    df = pd.DataFrame({"label": labels, "utterance": utterances})

    return df


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

    X_train, X_test, y_train, y_test = train_test_split(
        df["utterance"], df["label"], test_size=0.15, random_state=42
    )

    # Vectorized train & test utterances
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Baseline models need non-vectorized utterences, ML models need vectors so returning both types
    return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, le


def baseline_model_1(X_test):
    # Baseline model No. 1
    # Assume all utterances are "inform" - the most common label.

    # Create an array with length of X_test (all test cases) and fill with encoded version of label "inform"
    inform_encoding = le.transform(["inform"])[0]
    y_predicted = np.full(len(X_test), inform_encoding)

    return y_predicted


def baseline_model_2(X_test):
    # Baseline model No. 2
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
    keyWords["null"] = ["noise", "sil", "unintelligible", "cough"]
    keyWords["repeat"] = ["repeat", "that", "back", "again"]
    keyWords["reqalts"] = ["about", "how", "else"]
    keyWords["reqmore"] = ["more"]
    keyWords["request"] = ["the", "number", "phone", "postcode", "address"]
    keyWords["restart"] = ["start", "over", "restart"]
    keyWords["thankyou"] = ["thank"]

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


def assess_performance(y_test, y_predicted, model_name, loud=True):
    # Calculate metrics for the given model predictions
    if loud:
        print(f"Results for Model -> {model_name}")

    # Print different scores up to 2 decimal points (.2f)
    precision = precision_score(y_test, y_predicted, average='macro', zero_division=1)
    recall = recall_score(y_test, y_predicted, average='macro')
    f1_score =  2 * precision * recall / (precision + recall)
    if loud:
        print(f" Accuracy Score: {accuracy_score(y_test, y_predicted):.2f}")
        print(f" Precision Score: {precision:.2f}")
        print(f" Recall Score: {recall:.2f}")
        print(f" F1 Score: {f1_score:.2f}")


def predictions_process(df: pd.DataFrame):
    # Go through the prediction process for a given DataFrame

    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, _ = preprocess(df)

    y_baseline_1 = baseline_model_1(X_test)
    assess_performance(y_test, y_baseline_1, "Baseline 1")

    y_baseline_2 = baseline_model_2(X_test)
    assess_performance(y_test, y_baseline_2, "Baseline 2")

    # 1000 because with default (100) MAX_ITER warning was reached
    log_regression = LogisticRegression(max_iter=1000)
    log_regression.fit(X_train_vec, y_train)
    y_log_reg = log_regression.predict(X_test_vec)
    assess_performance(y_test, y_log_reg, "Logistic Regression")

    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=18)
    # TODO explain why 18
    # We did an analysis, calculated the number based on some formula.
    decision_tree = decision_tree.fit(X_train_vec, y_train)
    y_decision_tree = decision_tree.predict(X_test_vec)
    assess_performance(y_test, y_decision_tree, "Decision Tree")

    knn = KNeighborsClassifier(
        n_neighbors=147
    )  # Choose K according rule: sqrt(N) where N is number of instances
    knn.fit(X_train_vec, y_train)
    y_knn = knn.predict(X_test_vec)
    assess_performance(y_test, y_knn, "K-Nearest Neighbors")

    return log_regression, decision_tree, knn


def train_log_reg():
    # To be used ad-hoc, by the StateMachine
    df = load_file()
    df_deduplicated = (
        df.drop_duplicates()
    )  # We want to use the model trained on deduplicated data, as we believe it to generalize better
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, _ = preprocess(
        df_deduplicated
    )

    # Selected as the best model
    log_regression = LogisticRegression(max_iter=1000)
    log_regression.fit(X_train_vec, y_train)

    return log_regression


def train_knn():
    # To be used ad-hoc, by the StateMachine
    df = load_file()
    df_deduplicated = (
        df.drop_duplicates()
    )  # We want to use the model trained on deduplicated data, as we believe it to generalize better
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, _ = preprocess(
        df_deduplicated
    )

    knn = KNeighborsClassifier(
        n_neighbors=147
    )  # Choose K according rule: sqrt(N) where N is number of instances
    knn.fit(X_train_vec, y_train)

    return knn


def main():
    # Main function running the whole process and asking for user input
    if len(sys.argv) < 2:
        print("To specify a different input file use: python main.py <path_to_dialog-act.dat>")
        
    file_path = sys.argv[1]
    df = load_file(file_path)
    
    print("Loaded DataFrame:")
    print(df.head())

    df_full = df
    df_deduplicated = df.drop_duplicates()

    print("Full Dataset Predictions: ")
    print("-----------------------------------------------")

    # Take Logistic Regression as the best model for future predictions
    log_reg, decision_tree, knn = predictions_process(df_full)

    print("De-Duplicated Dataset Predictions: ")
    print("-----------------------------------------------")

    predictions_process(df_deduplicated)

    while True:
        custom_message = input(
            "Enter a custom message (or type 'exit' to quit): "
        ).lower()
        if custom_message == "exit":
            print("Done with predicting. Goodbye!")
            break
        else:
            # Vectorize the custom message
            custom_message_vec = vectorizer.transform([custom_message])

            prediction = log_reg.predict(custom_message_vec)
            prediction_1d = np.array(
                [prediction]
            ).ravel()  # Change shape to pacify a warning from LabelEncoder
            prediction_label = le.inverse_transform(prediction_1d)

            print(f"You entered: {custom_message}")
            print(
                f"Predicted Label using Logistic Regresssion is: {prediction_label[0]}"
            )

            prediction = decision_tree.predict(custom_message_vec)
            prediction_1d = np.array(
                [prediction]
            ).ravel()  # Change shape to pacify a warning from LabelEncoder
            prediction_label = le.inverse_transform(prediction_1d)

            print(f"You entered: {custom_message}")
            print(f"Predicted Label using Decision Tree is: {prediction_label[0]}")

            prediction = knn.predict(custom_message_vec)
            prediction_1d = np.array(
                [prediction]
            ).ravel()  # Change shape to pacify a warning from LabelEncoder
            prediction_label = le.inverse_transform(prediction_1d)

            print(f"You entered: {custom_message}")
            print(f"Predicted Label using KNN is: {prediction_label[0]}")


if __name__ == "__main__":
    main()


class Models:
    def __init__(self):
        log_reg = train_log_reg()
        knn = train_knn()

    def predict_dialog_act(utterance, model: str = "log_reg"):
        # Predict dialog act with a chosed model
        global LOG_REG_TRAINED, KNN_TRAINED
        if model == "log_reg" and LOG_REG_TRAINED is False:
            log_reg = train_log_reg()
            LOG_REG_TRAINED = True
        elif model == "knn" and KNN_TRAINED is False:
            knn = train_knn()
            KNN_TRAINED = True

        custom_message_vec = vectorizer.transform([utterance])
        if model == "log_reg":
            prediction = log_reg.predict(custom_message_vec)
        elif model == "knn":
            prediction = knn.predict(custom_message_vec)

        prediction_1d = np.array(
            [prediction]
        ).ravel()  # Change shape to pacify a warning from LabelEncoder
        prediction_label = le.inverse_transform(prediction_1d)
        print(f"PREDICTED: {prediction_label}")

        return prediction_label
