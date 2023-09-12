import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


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

  le = LabelEncoder() 
  # Fit the encoder with a list of all labels
  le.fit(df['label'].unique()) 
  # Replace the labels with encodings
  df['label'] = le.transform(df['label'])

  vectorizer = CountVectorizer(binary=True, strip_accents='unicode',
                                  max_features=90000)
  # Fit the vectorizer with utterances
  vectorizer = vectorizer.fit(df['utterance'])

  X_train, X_test, y_train, y_test = split_dataset(df)

  X_train_vec = vectorizer.transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, le


# Main function running the whole process and asking for user input
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_dialog-act.dat>")
        return

    file_path = sys.argv[1]
    df = load_data(file_path)
    
    print("Loaded DataFrame:")
    print(df.head())
    
    custom_message = input("Enter a custom message: ")
    print(f"You entered: {custom_message}")

if __name__ == "__main__":
    main()
   