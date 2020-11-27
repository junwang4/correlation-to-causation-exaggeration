
#short_cut: sentence_claim_strength_classification
#short_cut: observational_study
short_cut: exaggeration_analysis


###################################
data_type = eureka
data_type = pubmed

claim_clf_task = apply_one_full_model_to_new_sentences
claim_clf_task = train_one_full_model
claim_clf_task = evaluate_and_error_analysis
claim_clf_task = train_KFold_model

data_augmentation = False
data_augmentation = True

sentence_claim_strength_classification:
	python run.py sentence_claim_strength_classification --data_type=$(data_type) --data_augmentation=$(data_augmentation) --task=$(claim_clf_task)


###################################
obs_clf_task = train
obs_clf_task = predict_pubmed_observational_study_for_press_releases
obs_clf_task = test

observational_study:
	python run.py observational_study_classification --task=$(obs_clf_task)


###################################
exagg_task = plot_trend_of_exaggeration
exagg_task = plot_number_of_observational_studies_over_years
exagg_task = aggregate_from_sentence_to_article_level__and__merge_aggregated_pubmed_and_eureka
exagg_task = expand_the_aggregated_and_merged_data_with_ac_jo
exagg_task = plot_university_vs_journal

exaggeration_analysis:
	python run.py exaggeration_analysis --task=$(exagg_task)

