# Team 4 Report

## Project description

## Quantitative evaluation 
To evaluate the performance of the classifiers, we have selected a set of four key evaluation metrics, these metrics include Accuracy, Precision, Recall and F1 Score. These metrics help us understand how well our classifiers are performing on various aspects. See below a short motivation per metrics. 

Accuracy: this is a straightforward measure of overall correctness. With measuring the accuracy you answer the question of how many predictions our classifiers got right, out of all the predictions made. We aim for a high accuracy, as that is desirable because it indicates a high proportion of correct predictions. 

Precision: building in accuracy we also decided to measure the level of precision. Calculating this score gives you an insight in the proportion of true positive predictions out of all the positive predictions our classifiers have made. A high precision score is valuable to us as we want to minimize the false positive error. 

Recall: with measuring recall we capture our classifiers' ability to identify all the instances right. To be precise, with recall we calculate the exact proportion of true positive predictions among all the actual positive instances that occur in our dataset. This is important if we want to minimize the false negatives. 

F1 Score: the last metrics we have chosen is the F1 Score. The F1 score takes into account both the false positives and false negatives, reaching a balance. This we measure because we aim to achieve a balance between making accurate positive predictions and capturing all relevant positive instances. 

## Error analysis
*Are there specific dialog acts that are more difficult to classify? Are there particular utterances that are hard to classify (for all systems)? And why?*
...

The error analysis demonstrates that the three machine learning models all struggle the most with correctly classifying the dialog acts with the label "null". Dialog acts with the label “inform” and “reqalts” (request alternatives) also tend to be more difficult to classify for all models.
Utterances that are hard to classify often contain abbreviations, typos, or shorter spelling by for example leaving out vowels. Furthermore, some utterances are wrongly labeled in the dataset. By way of illustration, the utterance "what about kosher food" is labeled “inform”, while the label "reqalts" is more applicable. Lastly, the three machine learning models take the first labeled dialog act, while the utterances in the original dataset can have more than one labeled dialog act. As a results, the models may misclassify utterances based on the first labeled dialog act, even though there could be other dialog acts present in the original dataset that was not considered.


## Difficult cases
*Come up with two types of ‘difficult instances’, for example utterances that are not fluent (e.g. due to speech recognition issues) or the presence of negation (I don’t want an expensive restaurant). For each case, create test instances and evaluate how your systems perform on these cases.*
...
The following two difficult instances were tested on our models:
  1. "thx bai"
  2. "wht non veggie options u got"
  3. 
Both cases were labeled “inform” by the three implemented models. However, for the first utterance the dialog act "bye" should be applied, and for the second utterance the dialog act "request". Since the inform label is most present in the data, it could be an explanation that there is a tendency of assigning this label to the utterances.


## System comparison
*How do the systems compare against the baselines, and against each other? What is the influence of deduplication? Which one would you choose for your dialog system?*
...

Baseline 1 shows a high precision due to the model always predicting the majority label “inform”. As a result, the majority of the predictions are correct (true positives). The models' evaluation metrics demonstrate a low recall, because always predicting the majority leads to missing a significant number of positive cases. For the Baseline 2 model, the scores depend mainly on the added rules. There is again a relatively high precision and low recall, resulting in a not extremely high F1-score. The logistic regression model has a more balanced precision and recall score. This indicates that it correctly classifies utterances from majority and minority classes. The decision tree model also shows a better performance than the baseline models. A possible reason for this is that a decision tree model has the ability to learn from the data, in contrast to the two baselines. The KNN model performs better than the baseline models for the accuracy, precision and F1-score, however the recall is slightly better for the Baseline 2 model. Possible reasons for the low recall are that the KNN model is sensitive to class imbalance, and high dimensional data. 
Overall, the machine learning models outperform the simplistic baseline models. The logistic regression exceeds in performance, since it is less sensitive to high-dimensional data and class imbalance compared to the other two machine learning models.
Regarding the deduplicated dataset, all the models’ performance decreases except for baseline 1 that slightly increases in performance. Data leakage is a possible reason for the poor performance of the models. Removing duplicates causes the models to evaluate data that did not have the same characteristics as the data they were trained on. Another reason could be overfitting. It is possible that the models performed well on the training data (overfit), but struggled to generalize to unseen data. 
Taking all of this into account it is better to work with the deduplicated dataset for the initial development of the dialog system. Once the models perform well, we can consider incorporating more data.

