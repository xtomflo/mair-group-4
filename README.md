# mair-group-4
Group Project repository for Methods in AI Research course

Team:
Alex
Liliya
Isabelle
Dean
Tomek 


Code is split into 3 python files:
- dialog_machine.py     - main file implementing the dialog machine
- models.py             - base file with model definitions, baselines, and error analysis
- utils.py              - file with helper functions

Additional files:
- requirements.txt      - stores pip package requirements
- restaurant_info.csv   - stores the restaurant information with added columns on food_quality,crowdedness,length_of_stay
- dialog_acts.dat       - file with dialog_act required for training the classifiers.

# Instructions 
```
We're assuming Python 3.10 is installed, together with pip 23.0.1
```
1. Make sure you have the required libraries installed
```
pip3 install --requirement requirements.txt
```

2. Running the application

To start the recommender chatbot:
```
> python3 dialog_machine.py
```

To run model training, evaluation and error analysis:
```
> python3 models.py
```


