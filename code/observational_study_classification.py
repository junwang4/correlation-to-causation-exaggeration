import os, sys, re, json, datetime, time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import helpers
import lightgbm as lgb

from configparser import ConfigParser
config = ConfigParser()
config.read('settings.ini')
DATA_FILE_TO_WORK_PUBMED = config.get('common', "DATA_FILE_TO_WORK_PUBMED")
PREDICTED_OBSERVATIONAL_FILE_PUBMED = config.get('common', 'PREDICTED_OBSERVATIONAL_FILE_PUBMED')
TAG = config.get('common', "TAG")

class ObservationalStudyClassifier:
    def __init__(self):
        self.working_folder = helpers.get_or_create_dir("working/observational_study_classifier")
        self.data_folder = "../data"
        self.model_version = "20201102"
        #self.model_version = helpers.get_current_date_str()

    def get_fpath_of_pubmed_data_to_predict_observational_or_not(self):
        return f'{self.data_folder}/{DATA_FILE_TO_WORK_PUBMED}'

    def get_fpath_of_predicted_observational_studies(self):
        return f'{self.working_folder}/../{TAG}_{PREDICTED_OBSERVATIONAL_FILE_PUBMED}'

    def get_fpath_of_observational_studies_training_set(self, cls):
        return f'{self.data_folder}/observational_vs_trial_study_labeled_{cls}.csv'

    def get_fpath_of_observational_studies_classifier_model(self, mode_type):
        return f'{self.working_folder}/observational_or_trial_model_{mode_type}_{self.model_version}.m'

    def get_fpath_of_observational_studies_classifier_model_tifidf(self):
        return self.get_fpath_of_observational_studies_classifier_model('tifidf')

    def get_fpath_of_observational_studies_classifier_model_lgb(self):
        return self.get_fpath_of_observational_studies_classifier_model('lgb')


    def train(self):
        """
        :return: lgb.m, tfidf.m
        """
        self.train_and_test(mode='train')

    def test(self):
        self.train_and_test(mode='test')

    def train_and_test(self, mode='test'):
        fpath_obs = self.get_fpath_of_observational_studies_training_set(1)
        fpath_trial = self.get_fpath_of_observational_studies_training_set(0)
        df_obs = pd.read_csv(fpath_obs)
        df_trial = pd.read_csv(fpath_trial)

        def get_text(x):
            return x['title'] + ' ' + ' '.join([txt for label, txt in json.loads(x['abstract_json'])])

        df_obs['text'] = df_obs.apply(get_text, axis=1)
        df_trial['text'] = df_trial.apply(get_text, axis=1)

        df_train = pd.concat((df_obs, df_trial))
        X = df_train['text'].values
        y = df_train['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        if mode == 'train':
            print('\n- this may take about 10 minutes to train 50,000 examples\n')

            tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', min_df=3, ngram_range=(1,2))
            vectorizer = tfidf_vectorizer
            X_train_vec = vectorizer.fit_transform(X_train)
            d_train = lgb.Dataset(X_train_vec, label=y_train)

            params = {}
            params['learning_rate'] = 0.005
            params['boosting_type'] = 'gbdt'
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            params['feature_fraction'] = 0.3

            nrounds = 1500   # 1500: takes about 10 minutes for training 50,000
            nrounds = 3000
            #nrounds = 30  # 30: 100 seconds

            clf = lgb.train(params, d_train, nrounds)
            joblib.dump(vectorizer, self.get_fpath_of_observational_studies_classifier_model_tifidf())
            joblib.dump(clf, self.get_fpath_of_observational_studies_classifier_model_lgb())

        elif mode == 'test':
            vectorizer = joblib.load(self.get_fpath_of_observational_studies_classifier_model_tifidf())
            clf = joblib.load(self.get_fpath_of_observational_studies_classifier_model_lgb())

        X_test_vec = vectorizer.transform(X_test)
        print(X_test_vec.shape)

        y_pred_lgb = clf.predict(X_test_vec)

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_lgb)
        print(len(y_test), len(precision))

        prec = 0.93
        idx0 = np.argmax(precision>=prec)
        for idx in range(idx0, idx0+1000, 100):
            print(f'{idx}  recall: {recall[idx]:.3f}  precision: {precision[idx]:.4f}  threshold: {thresholds[idx]:.4f}' )

        print(y_test)
        print(y_test.shape)
        print(y_pred_lgb)
        print(y_pred_lgb.shape)
        y_pred = [int(x>=0.5) for x in y_pred_lgb]
        print(classification_report(y_test, y_pred, digits=3))


    def _predict_one_file(self, fpath, vectorizer, clf):
        df = pd.read_csv(fpath, encoding='latin-1', usecols='pmid title abstract_json'.split())
        df.fillna('', inplace=True)
        print('size:', len(df), fpath)

        def get_abstract_and_text(x):
            out = [ x['title'] ]
            abs = json.loads(x['abstract_json'])
            if abs:
                for subsection in abs:
                    label, text = subsection
                    out.append(text)
            try:
                return ' '.join(out)
            except:
                print('- error:')
                print(x)
                sys.exit()

        df['text'] = df.apply(get_abstract_and_text, axis=1)
        X_test = df['text'].values
        X_test_vec = vectorizer.transform(X_test.ravel())
        y_pred_proba = clf.predict(X_test_vec)

        df_test = pd.DataFrame()
        df_test['pred_proba'] = y_pred_proba
        df_test['pmid'] = df.pmid
        cols = 'pmid pred_proba'.split()
        return df_test[cols]

    def predict_pubmed_observational_study_for_press_releases(self):
        fpath_tfidf_model = self.get_fpath_of_observational_studies_classifier_model_tifidf()
        fpath_lgb_model = self.get_fpath_of_observational_studies_classifier_model_lgb()

        vectorizer = joblib.load(fpath_tfidf_model)
        clf = joblib.load(fpath_lgb_model)

        df = self._predict_one_file(self.get_fpath_of_pubmed_data_to_predict_observational_or_not(), vectorizer, clf)
        df.to_csv(self.get_fpath_of_predicted_observational_studies(), index=False, float_format='%.4f')
        print('- total studies:', len(df))
        for th in np.arange(0.5,1,0.05): print(f'- decision threshold={th:.2f}: {len(df[df.pred_proba>=th])}')


def main():
    oclf = ObservationalStudyClassifier()
    #oclf.train()
    #oclf.test()
    oclf.predict_pubmed_observational_study_for_press_releases()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')
