import os, sys, time, re, json
import numpy as np
import pandas as pd
from IPython.display import display
from configparser import ConfigParser
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from bert_sklearn import BertClassifier, load_model
import helpers

pd.options.display.max_colwidth = 80
pd.options.display.width = 1000
pd.options.display.precision = 3
np.set_printoptions(precision=3)

# if set to True, it requires the installation of https://github.com/junwang4/bert-sklearn-with-class-weight
USE_CLASS_WEIGHT_FOR_UNBALANCED_DATA = True

LABEL_NAME = {0:'none', 1:'corr', 2:'cond', 3:'causal'}
NUM_CLASSES = len(LABEL_NAME)

BERT_NAME_2_MODEL = {'bert' : 'bert-base-cased', 'biobert' : 'biobert-base-cased'}

config = ConfigParser()
config.read('settings.ini')

BERT_NAME = config.get('bert', 'BERT_NAME')
print('- bert type:', BERT_NAME)

K_FOLDS = config.getint('bert', 'K_FOLDS')
EPOCHS = config.getint('bert', 'EPOCHS')

MAX_SEQ_LENGTH = config.getint('bert', 'MAX_SEQ_LENGTH')
TRAIN_BATCH_SIZE = config.getint('bert', 'TRAIN_BATCH_SIZE')
LEARNING_RATE = config.getfloat('bert', 'LEARNING_RATE')

RANDOM_STATE = config.getint('bert', 'RANDOM_STATE')
SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD = config.getboolean('bert', 'SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD')

HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT = config.get('bert', "HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT")

ANNOTATED_FILE_PUBMED = config.get('bert', "ANNOTATED_FILE_PUBMED")
ANNOTATED_FILE_EUREKA = config.get('bert', "ANNOTATED_FILE_EUREKA")

DATA_FILE_TO_WORK_PUBMED = config.get('common', "DATA_FILE_TO_WORK_PUBMED")
DATA_FILE_TO_WORK_EUREKA = config.get('common', "DATA_FILE_TO_WORK_EUREKA")
TAG = config.get('common', "TAG")


class SentenceClassifier:
    def __init__(self, data_type='eureka', data_augmentation=False):
        self.task = data_type
        self.class_balance = USE_CLASS_WEIGHT_FOR_UNBALANCED_DATA
        self.kfolds = K_FOLDS
        self.bert_model_name = BERT_NAME
        self.bert_model = BERT_NAME_2_MODEL[self.bert_model_name]

        self.data_folder = '../data'
        self.working_folder = helpers.get_or_create_dir(f'working/{self.task}')
        self.model_folder = helpers.get_or_create_dir(f'{self.working_folder}/model_{self.task}_{self.bert_model_name}')
        self.pred_folder = helpers.get_or_create_dir(f'{self.working_folder}/pred_{self.task}_{self.bert_model_name}')

        if self.task=='eureka':
            self.annotated_file = ANNOTATED_FILE_EUREKA
            self.augmented_file = ANNOTATED_FILE_PUBMED
            self.data_file_to_work = DATA_FILE_TO_WORK_EUREKA
            self.data_augmentation = data_augmentation
        else:
            self.annotated_file = ANNOTATED_FILE_PUBMED
            self.data_file_to_work = DATA_FILE_TO_WORK_PUBMED
            self.data_augmentation = False

        print('\n- data_augmentation:', self.data_augmentation)

    def get_train_data_csv_fpath(self):
        fpath = f'{self.data_folder}/{self.annotated_file}'
        print('- annotated csv file:', fpath)
        if os.path.exists(fpath):
            return fpath
        else:
            print('- error: training csv file not exists:', fpath)
            sys.exit()

    def read_train_data(self):
        return pd.read_csv(self.get_train_data_csv_fpath(), usecols=['sentence', 'label'], encoding = 'utf8', keep_default_na=False)

    def get_class_weight(self, labels):
        class_weight = [x for x in compute_class_weight("balanced", range(len(set(labels))), labels)]
        print('- auto-computed class weight:', class_weight)
        return class_weight

    def get_model_bin_file(self, fold=0):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f'\ncreate a new folder for storing BERT model: "{self.model_folder}"\n')
        if fold>=0:
            return f'{self.model_folder}/K{self.kfolds}_epochs{EPOCHS}_{fold}.bin'
        elif fold==-1:
            return f'{self.model_folder}/full_epochs{EPOCHS}.bin'
        else:
            print('- wrong value for fold:', fold)
            sys.exit()

    def get_pred_csv_file(self, mode='train'):
        if mode == 'train':
            fpath = f'{self.pred_folder}/{mode}_K{self.kfolds}_epochs{EPOCHS}_augment{self.data_augmentation}.csv'
        elif mode == 'apply':
            fpath = f'{self.pred_folder}/{TAG}_{mode}_epochs{EPOCHS}.csv'
        else:
            print('- wrong mode:', mode, '\n')
            sys.exit()
        print('- read pred csv file:', fpath)
        return fpath

    def get_train_test_data(self, df, fold=0):
        df['sentence'] = df.sentence.apply(lambda x: x.strip())
        kf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=RANDOM_STATE)
        cv = kf.split(df.sentence, df.label)

        for i, (train_index, test_index) in enumerate(cv): #kf.split(df.sentence, df.label)):
            if i == fold:
                break
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        print(f"\nALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
        label_list = np.unique(train.label)
        return train, test, label_list

    def train_model(self, df_train, model_file_to_save, val_frac=0.1, class_weight=None):
        X_train = df_train['sentence']
        y_train = df_train['label']

        model = BertClassifier(bert_model=self.bert_model, random_state=RANDOM_STATE, \
                                class_weight=class_weight, max_seq_length=MAX_SEQ_LENGTH, \
                                train_batch_size=TRAIN_BATCH_SIZE, learning_rate=LEARNING_RATE, \
                                epochs=EPOCHS, validation_fraction=val_frac)
        print(model)
        model.fit(X_train, y_train)

        if model_file_to_save:
            model.save(model_file_to_save)
            print(f'\n- model saved to: {model_file_to_save}\n')
        return model

    def _augment_training_data_of_press_release_with_all_data_from_pubmed_annotations(self, train_data):
        print('\n=== AUGMENTATION ===\n- size of training data before augmentation:', len(train_data))
        train_data_ = pd.concat((train_data, train_data))
        df_pubmed = pd.read_csv( f'{self.data_folder}/{self.augmented_file}' )
        print('- size of pubmed annotation as augmented data:', len(df_pubmed))
        print('- size of training data after augmentation should be "2*len(train_data) + len(df_pubmed)":', 2*len(train_data)+len(df_pubmed))
        train_data = pd.concat((train_data_, df_pubmed))
        print('- size of training data after augmentation is:', len(train_data))
        print()
        return train_data

    def train_one_full_model(self):
        df_train = self.read_train_data()

        if self.task == 'eureka' and self.data_augmentation:
            df_train = self._augment_training_data_of_press_release_with_all_data_from_pubmed_annotations(df_train)

        if self.class_balance:
            class_weight = self.get_class_weight(df_train['label'])
        else:
            class_weight = None

        model_file_to_save = self.get_model_bin_file(fold=-1) # -1: for one full model
        val_frac = 0.0
        self.train_model(df_train, model_file_to_save, val_frac=val_frac, class_weight=class_weight)

    def train_KFold_model(self, do_train=True):
        df = self.read_train_data()
        print('- label value counts:')
        print(df.label.value_counts())

        y_test_all, y_pred_all = [], []
        results = []
        df_out_proba = None
        for fold in range(self.kfolds):
            train_data, test_data, label_list = self.get_train_test_data(df, fold)

            if self.task=='eureka' and self.data_augmentation:
                train_data = self._augment_training_data_of_press_release_with_all_data_from_pubmed_annotations(train_data)

            if SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD:
                model_file = self.get_model_bin_file(fold)
            else:
                model_file = ''

            class_weight = self.get_class_weight(df['label']) if self.class_balance else None

            val_frac = 0.05
            if do_train:
                model = self.train_model(train_data, model_file, val_frac=val_frac, class_weight=class_weight)
            else:
                model_file = self.get_model_bin_file(fold)
                model = load_model(model_file)

            X_test = test_data['sentence']
            y_test = test_data['label']
            y_test_all += y_test.tolist()

            y_proba = model.predict_proba(X_test)
            del model

            tmp = pd.DataFrame(data=y_proba, columns=[f'c{i}' for i in range(NUM_CLASSES)])
            tmp['confidence'] = tmp.max(axis=1)
            tmp['winner'] = tmp.idxmax(axis=1)
            tmp['sentence'] = X_test.tolist()
            tmp['label'] = y_test.tolist()
            df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))

            y_pred = [int(x[1]) for x in tmp['winner']]
            y_pred_all += y_pred

            acc = accuracy_score(y_pred, y_test)
            res = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

            item = {'Acc': acc, 'weight': len(test_data)/len(df), 'size': len(test_data)}
            item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
            for cls in np.unique(y_test):
                res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
                for i, scoring in enumerate('P R F1'.split()):
                    item['{}_{}'.format(scoring, cls)] = res[i][0]
            results.append(item)

            acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
            res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
            print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        # show an overview of the performance
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
        df_2['size'] = df_2['size'].astype(int)
        display(df_2)

        # put together the results of all 5-fold tests and save
        output_pred_csv_file_train = self.get_pred_csv_file(mode='train')
        df_out_proba.to_csv(output_pred_csv_file_train, index=False, float_format="%.3f")
        print(f'\noutput all {self.kfolds}-fold test results to: "{output_pred_csv_file_train}"\n')


    def apply_one_full_model_to_new_sentences(self):
        fpath_data = f'{self.data_folder}/{self.data_file_to_work}'

        df = pd.read_csv(fpath_data, nrows=None)

        if self.task == 'pubmed':
            #pmid,title,conclusion_sentences
            print('- cnt of pubmed papers:', len(df))
            data = [{'id':pmid, 'sentence':s} for pmid, ss in zip(df.pmid, df.conclusion_sentences) for s in ss.split('\t')]

        elif self.task == 'eureka':
            #eaid,pmid,date,contact,institution,title,summary,first_2_body_sentences
            print('- cnt of eureka press releases:', len(df))

            data = []
            for eaid, pmid, title, first_2_body_sentences in zip(df.eaid, df.pmid, df.title, df.first_2_body_sentences):
                try:
                    sentId = f'{pmid}__{eaid}__0__title'
                    data.append({'id':sentId, 'sentence':title})
                    for i in (1,2):
                        sentId = f'{pmid}__{eaid}__{i}__sentence'
                        data.append({'id':sentId, 'sentence':first_2_body_sentences.split('\t')[i-1]})
                except:
                    print(' - error with:', eaid, pmid)
                    print(' - title:', title)
                    print(' - first 2 sentences:', first_2_body_sentences)
                    print()

        # NOTE: df is replaced with new data
        df = pd.DataFrame(data)
        print('- cnt of sentences:', len(df))
        print(df[:5])

        print(f'all: {len(df):,}    unique sentences: {len(df.sentence.unique()):,}     papers or press releases: {len(df.id.unique()):,}')

        output_pred_file = self.get_pred_csv_file('apply')
        print(output_pred_file)

        model_file = self.get_model_bin_file(fold=-1)  # -1: indicating this is the model trained on all data
        print(f'\n- use trained model: {model_file}\n')

        model = load_model(model_file)

        model.eval_batch_size = 32
        y_prob = model.predict_proba(df.sentence)

        df_out = pd.DataFrame(data=y_prob, columns=[f'c{i}' for i in range(NUM_CLASSES)])
        df_out['confidence'] = df_out.max(axis=1)
        df_out['winner'] = df_out.idxmax(axis=1)
        for col in df.columns:
            df_out[col] = df[col]

        df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
        print(f'\n- output prediction to: {output_pred_file}\n')


    def check_performance_on_press_release_claim_classification_model_on_annotated_data(self):
        if self.task == 'pubmed':
            print('- error: this is for Eureka only')
            return

        fpath_data = self.get_train_data_csv_fpath()
        df_pred = pd.read_csv(self.get_pred_csv_file(mode='train'))
        #print(df_pred.head())  # c0,c1,c2,c3,confidence,winner,sentence,label

        df_ann = pd.read_csv(fpath_data) # sentId,sentence,label
        # 20824800-w-lss012611-0-title,Low socioeconomic status increases depression risk in rheumatoid arthritis patients,1
        #print(df_ann.head())

        df_ann['is_headline'] = df_ann.sentId.apply(lambda x:x.split('-')[-1]=='title')

        mode = 'first2sentences'
        mode = 'all'
        mode = 'headline'

        if mode == 'headline':
            df_ann = df_ann[df_ann.is_headline]
            print('- number of headlines:', len(df_ann))
        elif mode == 'first2sentences':
            df_ann = df_ann[~df_ann.is_headline]
            print('- number of first 2 sentences:', len(df_ann))
        else:
            print('- number of all sentences:', len(df_ann))

        debug = 0
        if debug:
            helpers.display_df_cnt(df_ann, 'sentId')
            helpers.display_df_cnt(df_ann, 'sentence')
            helpers.display_df_cnt(df_ann, 'label')

        cnt = len(df_ann)
        df = df_ann.merge(df_pred['sentence label confidence winner'.split()])
        df = df.drop_duplicates('sentence')
        cnt2 = len(df)
        print('- number after merging and drop_duplicates:', len(df))
        if cnt!=cnt2:
            print(f'- WARNING: the above two numbers are not the same: {cnt} != {cnt2}\n')
            print(df_ann.sentence.value_counts()[:3])

        df['pred'] = df.winner.apply(lambda x:int(x[1]))
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(df.label, df.pred))
        print(classification_report(df.label, df.pred, digits=3))


    def evaluate_and_error_analysis(self):
        df = pd.read_csv(self.get_pred_csv_file(mode='train')) # -2: a flag indicating putting together the results on all folds
        df['pred'] = df['winner'].apply(lambda x:int(x[1])) # from c0->0, c1->1, c2->2, c3->3

        print('\nConfusion Matrix:\n')
        cm = confusion_matrix(df.label, df.pred)
        print(cm)

        print('\n\nClassification Report:\n')
        print(classification_report(df.label, df.pred, digits=3))

        out = ["""
    <style>
        * {font-family:arial}
        body {width:900px;margin:auto}
        .wrong {color:red;}
        .hi1 {font-weight:bold}
    </style>
    <table cellpadding=10>
    """]

        row = f'<tr><th><th><th colspan=4>Predicted</tr>\n<tr><td><td>'
        label_name = LABEL_NAME
        for i in range(NUM_CLASSES):
            row += f"<th>{label_name[i]}"
        for i in range(NUM_CLASSES):
            row += f'''\n<tr>{'<th rowspan=4>Actual' if i==0 else ''}<th align=right>{label_name[i]}'''
            for j in range(NUM_CLASSES):
                row += f'''<td align=right><a href='#link{i}{j}'>{cm[i][j]}</a></td>'''
        out.append(row + "</table>")

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                row = f"<div id=link{i}{j}><h2>{label_name[i]} => {label_name[j]}</h2><table cellpadding=10>"
                row += f'<tr><th><th>Sentence<th>Label<th>{label_name[0]}<th>{label_name[1]}<th>{label_name[2]}<th>{label_name[3]}<th>mark</tr>'
                out.append(row)

                df_ = df[(df.label==i) & (df.pred==j)]
                df_ = df_.sort_values('confidence', ascending=False)

                cnt = 0
                for c0, c1, c2, c3, sentence, label, pred in zip(df_.c0, df_.c1, df_.c2, df_.c3, df_.sentence, df_.label, df_.pred):
                    cnt += 1
                    mark = "" if label == pred else "<span class=wrong>oops</span>"
                    item = f"""<tr><th valign=top>{cnt}.
                            <td valign=top width=70%>{sentence}
                            <td valign=top>{label_name[label]}
                            <td valign=top class=hi{int(c0>max(c1,c2,c3))}>{c0:.2f}
                            <td valign=top class=hi{int(c1>max(c0,c2,c3))}>{c1:.2f}
                            <td valign=top class=hi{int(c2>max(c0,c1,c3))}>{c2:.2f}
                            <td valign=top class=hi{int(c3>max(c0,c1,c2))}>{c3:.2f}
                            <td valign=top>{mark}</tr>"""
                    out.append(item)
                out.append('</table></div>')

        fpath_out = f'{helpers.get_or_create_dir(HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT)}/error_analysis_{self.task}.html'
        with open(fpath_out, 'w') as fout:
            fout.write('\n'.join(out))
            print(f'\n- Error analysis result saved to: "{fpath_out}"\n')


def main():
    clf = SentenceClassifier(data_type='eureka')
    clf = SentenceClassifier(data_type='eureka', data_augmentation=True)
    clf = SentenceClassifier(data_type='pubmed')

    #clf.train_KFold_model()
    #clf.evaluate_and_error_analysis()

    #clf.train_one_full_model()
    clf.apply_one_full_model_to_new_sentences()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')

