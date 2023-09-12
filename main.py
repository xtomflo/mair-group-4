import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Encoder and Vectorizer are used globally
le = LabelEncoder() 
vectorizer = CountVectorizer(binary=True, strip_accents='unicode',
                                    max_features=90000)

def load_file(file_path:str = "dialog_acts.dat") -> pd.DataFrame:

    # Initialize empty lists to store labels and utterances
    labels = []
    utterances = []

    # Read the file line by line

    with open(file_path, "r") as f:
        for line in f:
            # Split each line into label and utterance
            parts = line.strip().split(" ", 1)
            label, utterance = parts
            labels.append(label)
            utterances.append(utterance)

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'label': labels,
        'utterance': utterances
    })

    return df


# Converts all the columns in the DataFrame to lowercase
def convert_to_lowercase(df:pd.DataFrame):

    df.label = df.label.str.lower()
    df.utterance = df.utterance.str.lower()

    return df


def split_dataset(df:pd.DataFrame):

    X_train, X_test, y_train, y_test = train_test_split(df['utterance'], df['label'], test_size=0.15, random_state=42)

    return (X_train, X_test, y_train, y_test)


def preprocess(df:pd.DataFrame):
# Preprocessing:
# 1. Encode Labels
# 2. Split into test & training sets
# 3. Embed Sentences to Vectors 

    # Fit the encoder with a list of all labels
    le.fit(df['label'].unique()) 
    # Replace the labels with encodings
    df['label'] = le.transform(df['label'])

    # Fit the vectorizer with utterances
    vectorizer.fit(df['utterance'])

    X_train, X_test, y_train, y_test = split_dataset(df)

    # Vectorized train & test utterances
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Baseline models need non-vectorized utterences, ML models need vectors
    return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec


def baseline_model_1(X_test):
    # Baseline model No. 1
    # Assume all utterances are "inform" - the most common label.
    
    # Create an array with length of X_test (all test cases) and fill with label "inform", encoded as 6  
    inform_encoding = le.transform(['inform'])[0]
    y_predicted = np.full(len(X_test), inform_encoding)

    return y_predicted

def baseline_model_2(X_test):
    return 
    
    
def train_logistic_regression(X_train_vec, y_train):

    # 1000 because with default (100) MAX_ITER warning was reached
    classifier = LogisticRegression(max_iter = 1000)
    classifier.fit(X_train_vec, y_train)


    return classifier

def assess_performance(y_test, y_predicted, model_name):
    
    print(f"Results for Model -> {model_name}")
    
    # Print different scores up to 2 decimal points (.2f)
    print(f" Accuracy Score: {accuracy_score(y_test, y_predicted):.2f}")
    print(f" Precision Score: {precision_score(y_test, y_predicted, average='weighted', zero_division=1):.2f}")
    print(f" Recall Score: {recall_score(y_test, y_predicted, average='weighted'):.2f}")
    print(f" F1 Score: {f1_score(y_test, y_predicted, average='weighted'):.2f}")



def predictions_process(df:pd.DataFrame):
    
    X_train, X_test, y_train, y_test, X_train_vec, X_test_vec = preprocess(df)
    
    
    y_baseline_1 = baseline_model_1(X_test)
    assess_performance(y_test, y_baseline_1, "Baseline 1")
    
    log_reg = train_logistic_regression(X_train_vec, y_train)
    y_log_reg = log_reg.predict(X_test_vec)
    assess_performance(y_test, y_log_reg, "Logistic Regression")
    
    
def main():
    # Main function running the whole process and asking for user input
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_dialog-act.dat>")
        return

    file_path = sys.argv[1]
    df = load_file(file_path)
    
    print("Loaded DataFrame:")
    print(df.head())

    df = convert_to_lowercase(df)
    
    df_deduplicated = df.drop_duplicates()
    df_full = df

    print("Full Dataset Predictions: ")
    print("-----------------------------------------------")

    predictions_process(df_full)
    
    print("De-Duplicated Dataset Predictions: ")
    print("-----------------------------------------------")
    
    predictions_process(df_deduplicated)

    
    while True:
        custom_message = input("Enter a custom message (or type 'exit' to quit): ").lower()
        if custom_message == 'exit':
            print("Done with predicting. Goodbye!")
            break
        else:
            # Vectorize the custom message
            custom_message_vec = vectorizer.transform([custom_message])
            
            prediction = log_reg.predict(custom_message_vec)
            prediction_1d = np.array([prediction]).ravel() # Change shape to pacify a warning from LabelEncoder
            prediction_label = le.inverse_transform(prediction_1d)
            
            print(f"You entered: {custom_message}")
            print(f"Predicted Label is: {prediction_label[0]}")
    
    
if __name__ == "__main__":
    main()
   