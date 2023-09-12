# mair-group-4
Group Project repository for Methods in AI Research course

Team:
Alex
Lilya
Isabelle
Dean
Tomek 


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
    - [x]LogisticRegression
    - ??
4. Create a Prompt to enter a new sentence and classify this sentence and repeat the prompt until the user exits.
    
## TO DO:



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