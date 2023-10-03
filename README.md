# mair-group-4
Group Project repository for Methods in AI Research course

Team:
Alex
Liliya
Isabelle
Dean
Tomek 


Week 2 Deliverables
The state transition diagram in graphical form with numbered states
A working dialog system interface that prints system utterances to the screen and processes user input utterances entered with the keyboard, implementing a state transition function corresponding to the diagram, using predictions from the classifier built in Part 1a as part of the input for state transitions

An algorithm identifying user preference statements in the sentences using pattern matching on variable keywords and value keywords on utterances classified as inform, using Levenshtein edit distance if necessary
A lookup function that retrieves suitable restaurant suggestions from the CSV database matching the preferences as extracted in the implemented algorithm


1. Design of the State Transitions with numbered states - Diagram 
2. Dialog system interface:
    - [ ] Pattern Matching Algorithm - 
    - [ ] Keyword Matching 
        Find the nearest matching keyword, from different categories
    - [ ] Find a matching restaurant an retrieve it
    - [ ] State Transitions depending on the current state, dialog_act, what's found in the utterance

# TODO: 
- [ ] Showing alernatives - restaurants with 2 matching criteria
- [ ] Showing alternatives - When keyword not found, show available ones (cheap, moderate, expensive | east, north, south etc.)
    If we can't understand users preference. 
- [ ] For food type - show nearest matching food type? 
- [ ] Expanding the states to Wait, Request Info
- [ ] Handle GOODBYE from everywhere.
- [ ] Handling for Individial Info requests - Separate Address, PostCode, Phone. 
- [ ] Handling for WORLD food (test cases) - should be international? 
- [ ] Handling for DONT CARE type of preferences in each of the category. 



Week 1 Deliverables

1. Preprocessing of data:
    - Create two versions of the dataset
        - [x] As is.
        - [x] With duplicates removed
    - [x] Convert the data to lower case for training and testing
    - [x] Split into training & test 85% to 15%
2. Build two baseline systems
    - [x] regardless of the content of the utterance, always assigns the majority class of in the data. ( Inform ) which is 40% 
    - [ ] baseline rule-based system based on keyword matching. An example rule could be: anytime an utterance contains ‘goodbye’, it would be classified with the dialog act bye
3. Create classifiers 
    - [x] LogisticRegression
    - [x] KNN
    - [x] DecisionTree
4. Create a Prompt to enter a new sentence and classify this sentence and repeat the prompt until the user exits.
    - [x] done and using logistic regression to predict
    - [ ] predicting with Another Classifier?
    
# TO DO:

    - Baseline Model 2
    - Implement Another Classifier
    - Report?  We have first results which can be added there. 
        - Reasoning for why the de-duplicated have lower score? 
        - Why we get such good results with logistic regression? ¯\_(ツ)_/¯ 
        - Others? 
## Current Results - 12.09.23
```
Full Dataset Predictions: 
-----------------------------------------------
Results for Model -> Baseline 1
 Accuracy Score: 0.40
 Precision Score: 0.76
 Recall Score: 0.40
 F1 Score: 0.23
Results for Model -> Logistic Regression
 Accuracy Score: 0.98
 Precision Score: 0.98
 Recall Score: 0.98
 F1 Score: 0.98

De-Duplicated Dataset Predictions: 
-----------------------------------------------
Results for Model -> Baseline 1
 Accuracy Score: 0.56
 Precision Score: 0.75
 Recall Score: 0.56
 F1 Score: 0.40
Results for Model -> Logistic Regression
 Accuracy Score: 0.90
 Precision Score: 0.89
 Recall Score: 0.90
 F1 Score: 0.89
```
# Instructions 
```
We're assuming Python 3.10 is installed, together with pip 23.0.1
```
1. Make sure you have the required libraries installed
```
pip3 install --requirement requirements.txt
```

2. To run the program navigate to this projects folder in the terminal and run:
```
> python load_dialog_act.py dialog-act.dat
```
