# Measuring Correlation-to-Causation Exaggeration in Press Releases

This is a collaborative project with the School of Information at Syracuse University. 

**ABSTRACT**

Press releases have an increasingly strong influence on media coverage of health research;
however, they have been found to contain seriously exaggerated claims that can misinform the public and undermine public trust in science.
In this study we propose an NLP approach to identify exaggerated causal claims made in health press releases 
that report on observational studies, which are designed to establish correlational findings,
but are often exaggerated as causal. 
We developed a new corpus and trained models that can identify causal claims in the main statements in a press release.
By comparing the claims made in a press release with the corresponding claims in the original research paper, 
we found that 22% of press releases made exaggerated causal claims from correlational findings in observational studies.
Furthermore, universities exaggerated more often than journal publishers by a ratio of 1.5 to 1.
Encouragingly, the exaggeration rate has slightly decreased over the past 10 years,
despite the increase of the total number of press releases. 
More research is needed to understand the cause of the decreasing pattern.


### How to cite ###
Bei Yu, Jun Wang, Lu Guo, and Yingya Li (2020). 
Measuring Correlation-to-Causation Exaggeration in Press Releases. COLING'2020, pages 4860–4872,
December 8-13, 2020, online.
[PDF](https://www.aclweb.org/anthology/2020.coling-main.427.pdf)

```
@inproceedings{yu2020exaggerationCOLING,
  title={Measuring Correlation-to-Causation Exaggeration in Press Releases},
  author={Yu, Bei and Wang, Jun and Guo, Lu and Li, Yingya},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics (COLING'2020)},
  pages={4860-4872},
  year={2020},
  url={https://www.aclweb.org/anthology/2020.coling-main.427/}
}
```

Bei Yu, Yingya Li and Jun Wang (2019).
Detecting Causal Language Use in Science Findings. EMNLP'2019, pages 4656–4666, Hong Kong, China, November 3–7, 2019. 
[PDF](https://www.aclweb.org/anthology/D19-1473.pdf)

```
@inproceedings{yu2019EMNLPCausalLanguage,
  title={Detecting Causal Language Use in Science Findings},
  author={Yu, Bei and Li, Yingya and Wang, Jun},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4656-4666},
  year={2019},
  url={https://www.aclweb.org/anthology/D19-1473.pdf}
}
```

## Get started

### STEP 0. Prerequisite

My environment is Linux box (Ubuntu 16.04) with a 1080Ti GPU.

```
pip install fire
pip install lightgbm
```

Install bert-sklearn
https://github.com/junwang4/bert-sklearn-with-class-weight
(for handling imbalanced classes)
```
git clone https://github.com/junwang4/bert-sklearn-with-class-weight
cd bert-sklearn-with-class-weight
pip install .
```


### STEP 1. Get the repo of this project from github

```
git clone https://github.com/junwang4/correlation-to-causation-exaggeration
cd correlation-to-causation-exaggeration
ls -l
    code 
        Makefile  # make it easy to run the following code such as "python run.py ..."
        run.py
        sentence_claim_strength_classification.py
        exaggeration_analysis.py
        observational_study_classification.py
        helpers.py
        settings.ini
    data
        annotated_pubmed.csv
        annotated_eureka.csv
        20683_pubmed_conclusion_sentences.csv
        21342_eureka_beginning_sentences.csv
        observational_vs_trial_study_labeled_0.csv
        observational_vs_trial_study_labeled_1.csv
    README.md

cd code
```

### STEP 2. For PubMed research papers

#### 2.1 Train a fine-tuned BERT model for use in classifying the claim strength of the conclusion sentences in a paper abstract

```
python run.py sentence_claim_strength_classification --data_type=pubmed --data_augmentation=False --task=train_one_full_model
```
This will take as input an annotated dataset `data/annotated_pubmed.csv`,
and output a BERT model in folder 
`code/working/pubmed/model_pubmed_biobert`

#### 2.2 Apply the above trained model to the conclusion sentences extracted from the 
structured abstracts of 20,683 observational studies

```
python run.py sentence_claim_strength_classification --data_type=pubmed --data_augmentation=False --task=apply_one_full_model_to_new_sentences
```
This will take as input the file `data/20683_pubmed_conclusion_sentences.csv`,
and output a prediction csv file in folder 
 `code/working/pubmed/pred_pubmed_biobert`

#### 2.3 To evaluate the performance of the model, say, 5-fold cross-validation 

First, generate a prediction file as the result of training and testing each of the 5 folds
```
python run.py sentence_claim_strength_classification --data_type=pubmed --data_augmentation=False --task=train_KFold_model
```

Second, display the evaluation results, including:
- a classification report
- a error analysis report, saved in an HTML file located at 
`code/working/html`

```
python run.py sentence_claim_strength_classification --data_type=pubmed --data_augmentation=False --task=evaluate_and_error_analysis
```
In the case of using the default setting given in file `code/settings.ini`,
we have the following result:
```
              precision    recall  f1-score   support

           0      0.930     0.886     0.908      1356
           1      0.901     0.917     0.909       998
           2      0.766     0.906     0.830       213
           3      0.857     0.870     0.863       494

    accuracy                          0.895      3061
   macro avg      0.864     0.895     0.878      3061
weighted avg      0.898     0.895     0.896      3061
```

### STEP 3. For EurekAlert press releases

#### 3.1 Train a fine-tuned BERT model for use in classifying the claim strength of the headline and the 1st two sentences in a press release

```
python run.py sentence_claim_strength_classification --data_type=eureka --data_augmentation=True --task=train_one_full_model
```
This will take as input an annotated dataset `data/annotated_eureka.csv`,
and output a BERT model in folder 
`code/working/eureka/model_pubmed_biobert`

#### 3.2 Apply the above trained model to the headline and the 1st two sentences in the 21,342 press releases that reported the above 20,683 papers

```
python run.py sentence_claim_strength_classification --data_type=eureka --data_augmentation=True --task=apply_one_full_model_to_new_sentences
```
This will take as input the file `data/21342_eureka_beginning_sentences.csv`,
and output a prediction csv file in folder 
 `code/working/eureka/pred_eureka_biobert`

#### 3.3 To evaluate the performance of the model, say, 5-fold cross-validation 

First, generate a prediction file as the result of training and testing each of the 5 folds
```
python run.py sentence_claim_strength_classification --data_type=eureka --data_augmentation=True --task=train_KFold_model
```

Second, display the evaluation results, including:
- a classification report
- a error analysis report, saved in an HTML file located at 
`code/working/html`

```
python run.py sentence_claim_strength_classification --data_type=eureka --data_augmentation=True --task=evaluate_and_error_analysis
```
In the case of using the default setting given in file `code/settings.ini`,
we have the following result:
```
              precision    recall  f1-score   support

           0      0.829     0.825     0.827       486
           1      0.924     0.883     0.903       738
           2      0.919     0.954     0.936       284
           3      0.885     0.921     0.903       568

    accuracy                          0.890      2076
   macro avg      0.889     0.896     0.892      2076
weighted avg      0.890     0.890     0.890      2076
```


### STEP 4. Classify a PubMed paper as an observational study or not

#### 4.1 Train and test a LightGBM model

```
python run.py observational_study_classification --task=train
python run.py observational_study_classification --task=test
```

#### 4.2 Apply the trained model to the above 20,683 pubmed abstracts to find those observational studies
```
python run.py observational_study_classification --task=predict_pubmed_observational_study_for_press_releases
```


### STEP 5. Exaggeration analysis

#### 5.1 Aggregate from sentence-level to article-level for both PubMed and Eureka, as well as merge the aggregated PubMed and Eureka results
```
python run.py exaggeration_analysis --task=aggregate_from_sentence_to_article_level__and__merge_aggregated_pubmed_and_eureka
```
Output (depending on setting)
`working/COLING2020_aggregated_result_unanimous_three_claimConfidenceTH0.5_obsConfidenceTH0.5.csv`


#### 5.2 Generate three plots (as shown in Figure 3-a, 3-b and Figure 4 of our COLING'2020 paper), based on the above aggregated/merged data

```
python run.py exaggeration_analysis --task=plot_trend_of_exaggeration
python run.py exaggeration_analysis --task=plot_number_of_observational_studies_over_years
python run.py exaggeration_analysis --task=plot_university_vs_journal
```
Output folder: `working/figure/???.pdf`



## More about the data mentioned above
### PubMed research papers (specifically, the part of conclusion subsection in their structured abstracts)

__Annotated corpus__ 

`./data/annotated_pubmed.csv`

<table>
<tr><th>sentence</th><th>label</th></tr>
<tr><td>Levels of cholesterol fractions in patients with JIA were found within the normal range.</td><td>0</td></tr>
<tr><td>DM is associated with poor outcomes in patients undergoing hepatectomy.</td><td>1</td></tr>
<tr><td>Omega-3 fatty acids supplementation could elevate serum irisin in male patients with CAD.</td><td>2</td></tr>
<tr><td>DM and metabolic syndrome (MetS) have negative influence on fertility.</td><td>3</td></tr>
</table>

Meaning of label:

```
label = 0 : Not claim
label = 1 : Correlational
label = 2 : Conditional causal
label = 3 : Direct causal
```

__Conclusion sentences in the abstract of the 20,683 observational studies__

`./data/20683_pubmed_conclusion_sentences.csv`

<table>
<tr><th>pmid</th><th>conclusion_sentences</th></tr>
<tr><td>26599472</td><td>Disruption of language network structural hubs is directly associated with aphasia severity after stroke.</td></tr>
</table>
NOTE: If there are two or more sentences in `conclusion_sentences`, they are separated by a tab "\t".

 
### EurekAlert press releases
__Annotated corpus__  

`./data/annotated_eureka.csv`

<table>
<tr><th>sentence</th><th>label</th></tr>
<tr><td>Shave biopsy is a safe and acceptable method for initial evaluation of melanoma</td><td>0</td></tr>
<tr><td>Medically underserved girls receive less frequent evaluation for short stature.</td><td>1</td></tr>
<tr><td>Depression may increase the risk of kidney failure</td><td>2</td></tr>
<tr><td>Low socioeconomic status increases depression risk in rheumatoid arthritis patients</td><td>3</td></tr>
</table>

Meaning of label (same as above):

```
label = 0 : Not claim
label = 1 : Correlational
label = 2 : Conditional causal
label = 3 : Direct causal
```

__headline and first two sentences in the 21,342 press releases__

`./data/21342_eureka_beginning_sentences.csv`

<table>
<tr><th>eaid</th><th>pmid</th><th>date</th><th>contact</th><th>institution</th>
<th>title</th><th>first_2_body_sentences</th>
</tr>
<tr>
<td>2008-06/w-iio060308.php</td>
<td>18512713</td>
<td>2008-06-03</td>
<td>wiley.com</td>
<td>Wiley</td>
<td>Increased incidence of melanoma found in rheumatoid arthritis patients treated with methotrexate</td>
<td>A chronic, inflammatory disease of unknown origin, rheumatoid arthritis (RA) affects about 1 percent of adults worldwide.    Marked by joint destruction, RA often leads to disability and diminished quality of life.</td>
</tr>
</table>
NOTE: The two sentences in `first_2_body_sentences` are separated by a tab "\t".

### 25,000 title/abstracts of observational studies vs. 25,000 title/abstracts of trials

`./data/observational_vs_trial_study_labeled_1.csv`  # observational studies

`./data/observational_vs_trial_study_labeled_0.csv` # clinical trials

<table>
<tr><th>pmid</th><th>title</th><th>abstract_json</th><th>label</th></tr>
<tr>
<td>28972652</td><td>Shared decision-making for people with asthma.</td>
<td>[["background", "Asthma is a chronic ...</td>
<td>0</td>
</tr>
</table>
