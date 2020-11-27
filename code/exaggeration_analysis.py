import os, sys, re, json, time, random, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import helpers

from configparser import ConfigParser
config = ConfigParser()
config.read('settings.ini')

DATA_FILE_TO_WORK_PUBMED = config.get('common', "DATA_FILE_TO_WORK_PUBMED")
DATA_FILE_TO_WORK_EUREKA = config.get('common', "DATA_FILE_TO_WORK_EUREKA")
TAG = config.get('common', "TAG")

PREDICTED_OBSERVATIONAL_FILE_PUBMED = config.get('common', 'PREDICTED_OBSERVATIONAL_FILE_PUBMED')

AGGREGATION_STRATEGY = config.get('analysis', "aggregation_strategy")
HEADLINE_OR_THREE = config.get('analysis', "headline_or_three")
USE_CAUSAL2CAUSAL_EXAGGERATION = config.getboolean('analysis', "use_causal2causal_exaggeration")

claim_confidence_TH = config.getfloat('analysis', "claim_confidence_TH")
observational_study_prediction_confidence_TH = config.getfloat('analysis', "observational_study_prediction_confidence_TH")

pd.options.display.max_colwidth = 50
pd.options.display.width = 1000
pd.options.display.precision = 3
np.set_printoptions(precision=3)
plt.rc('text', usetex=True)

CLS_NAME = {'c0': 'None', 'c1':'correlational', 'c2':'conditional', 'c3':'causal'}
CLAIMS = ('correlational', 'conditional', 'causal')

def winner2string(winner):
    if winner not in CLS_NAME:
        print(f'- winner class "{winner}" should be in the list of {list(CLS_NAME)}')
        sys.exit()
    return CLS_NAME[winner]

class ExaggerationAnalysis:
    def __init__(self):
        self.data_folder = '../data'
        self.working_folder = 'working'

        self.claim_confidence_TH = claim_confidence_TH
        self.observational_study_prediction_confidence_TH = observational_study_prediction_confidence_TH

        self.combined_approach = AGGREGATION_STRATEGY
        self.headline_or_three = HEADLINE_OR_THREE
        self.use_causal2causal_exaggeration = USE_CAUSAL2CAUSAL_EXAGGERATION

        self.version = 'full' if self.use_causal2causal_exaggeration else 'pm_correlation_only'

    def _get_fpath_of_aggregated_pm_and_pr(self):
        return f'{self.working_folder}/{TAG}_aggregated_result_{self.combined_approach}_{self.headline_or_three}_claimConfidenceTH{self.claim_confidence_TH}_obsConfidenceTH{self.observational_study_prediction_confidence_TH}.csv'

    #def _get_fpath_data_for_github_repo_for_coling2020_submission(self):
    #    return f'{self.working_folder}/final_claimConfidenceTH{self.claim_confidence_TH}_obsConfidenceTH{self.observational_study_prediction_confidence_TH}.csv'

    def _get_fpath_of_eureka_title_and_first_two_sentence_predicted_for_claim_strength(self):
        return f'{self.working_folder}/eureka/pred_eureka_biobert/{TAG}_apply_epochs3.csv'

    def _get_fpath_of_pubmed_observational_conclusion_sentences_predicted_for_claim_strength(self):
        return f'{self.working_folder}/pubmed/pred_pubmed_biobert/{TAG}_apply_epochs3.csv'


    def _load_sentence_level_claim_strength_prediction_data_for_both_pubmed_and_eureka(self, nrows=None):
        fpath_claim_pubmed = self._get_fpath_of_pubmed_observational_conclusion_sentences_predicted_for_claim_strength()
        if not os.path.exists(fpath_claim_pubmed):
            print(f'\n- Error: file "{fpath_claim_pubmed}" not found')
            print('- To create the file, run the following command\n')
            print('  python run.py sentence_claim_strength_classification --data_type=pubmed --task=apply_one_full_model_to_new_sentences\n')
            sys.exit()

        df_pm = pd.read_csv(fpath_claim_pubmed, nrows=nrows) # pmid,sentence,winner(c0/c1/c2/c3)
        # c0,c1,c2,c3,confidence,winner,id,sentence
        df_pm.rename(columns={'id':'pmid'}, inplace=True)
        df_pm['pmid'] = df_pm['pmid'].apply(lambda x: str(int(x))) # to match with df_pr whose pmid is str
        df_pm.winner = df_pm.winner.apply(winner2string)
        helpers.display_first(df_pm, title='pubmed paper sentences')

        fpath_claim_eureka = self._get_fpath_of_eureka_title_and_first_two_sentence_predicted_for_claim_strength()
        if not os.path.exists(fpath_claim_eureka):
            print(f'\n- Error: file "{fpath_claim_eureka}" not found')
            print('- To create the file, run the following command\n')
            print('  python run.py sentence_claim_strength_classification --data_type=eureka --task=apply_one_full_model_to_new_sentences\n')
            sys.exit()

        df_pr = pd.read_csv(fpath_claim_eureka, nrows=nrows) # sentId,sentence,winner(c0/c1/c2/c3)
        # c0,c1,c2,c3,confidence,winner,id,sentence
        df_pr.rename(columns={'id':'sentId'}, inplace=True)
        df_pr.winner = df_pr.winner.apply(winner2string)
        # sentId: 26341461__2015-09/aaoo-drs083115.php__0__title
        df_pr['pmid'] = df_pr.sentId.apply(lambda x: x.split('__')[0])
        df_pr['eaid'] = df_pr.sentId.apply(lambda x: x.split('__')[1])

        if self.headline_or_three == 'headline':
            df_pr['sentId_'] = df_pr.sentId.apply(lambda x:x.split('.php')[0])
            df_pr = df_pr.sort_values('sentId').drop_duplicates('sentId_', keep='first')
            df_pr.drop('sentId_', inplace=True, axis=1)

        helpers.display_first(df_pr, title='press release sentences')
        return df_pm, df_pr


    def aggregate_from_sentence_to_article_level__and__merge_aggregated_pubmed_and_eureka(self):
        claim_confidence_TH = self.claim_confidence_TH
        def get_strongest_claim(winner_list):
            for clm in ['causal', 'conditional', 'correlational']:
                if clm in set(winner_list):
                    return clm
            return 'None'

        def get_most_likely_claim(winner_list, confidence_list):
            claim, confidence = 'None', 0
            for clm_, proba_ in zip(winner_list, confidence_list):
                if clm_ != 'None' and proba_ > confidence:
                    claim = clm_
                    confidence = proba_
            return claim

        def get_uanimous_claim(winner_list, confidence_list):
            confidence_list = [c for c in confidence_list]
            claims = set([winner for winner, proba in zip(winner_list, confidence_list) if winner!='None' and proba>=claim_confidence_TH])
            if len(claims) == 1:
                return list(claims)[0]
            else:
                return 'na'

        def get_fine_grained_claim_for_pr(winner_list, confidence_list):
            non_causal_set = ('correlational', 'conditional')
            ws = [w for w in winner_list]
            confidence_list = [c for c in confidence_list]
            if len(ws) < 3:
                print('- weird case (less than 3 sentences):', ws, confidence_list)
                return 'na'
            if ws[0]=='causal' and confidence_list[0]>=claim_confidence_TH:
                return 'causal'
            elif ws[0] in non_causal_set and confidence_list[0]>=claim_confidence_TH:
                return 'non_causal'
            elif ws[1] in non_causal_set and confidence_list[1]>=claim_confidence_TH:
                return 'non_causal'
            elif ws[2] in non_causal_set and confidence_list[2]>=claim_confidence_TH:
                return 'non_causal'
            elif ws[1]=='causal' and confidence_list[1]>=claim_confidence_TH:
                return 'causal'
            elif ws[2]=='causal' and confidence_list[2]>=claim_confidence_TH:
                return 'causal'
            else:
                return 'na'

        def concat_all_claims_and_strength(x):
            return ' // '.join([f'{winner} {proba:.3f}' for winner, proba in zip(x['winner'], x['confidence'])])

        def my_aggr(x, fine_grained=False):
            cols = ['combined', 'combined_full', 'paragraph']
            para = ' // '.join(x['sentence'])
            combined_full = concat_all_claims_and_strength(x)
            if self.combined_approach == 'most_likely':
                combined = get_most_likely_claim(x['winner'], x['confidence'])
            elif self.combined_approach == 'unanimous':
                if fine_grained:
                    combined = get_fine_grained_claim_for_pr(x['winner'], x['confidence'])
                else:
                    combined = get_uanimous_claim(x['winner'], x['confidence'])
            return pd.Series([combined, combined_full, para], index=cols)

        def aggregate_to_article_level_for_pr(df_pr):
            if self.headline_or_three == 'three':
                my_aggr_pr = lambda x: my_aggr(x, fine_grained=True)
            else:
                my_aggr_pr = my_aggr

            df_pr_grouped = df_pr.groupby(['eaid', 'pmid']).apply(my_aggr_pr).reset_index()

            df_pr_grouped['year'] = df_pr_grouped.eaid.apply(lambda x: int(x[:4]))

            df_pr_grouped.rename(columns={'paragraph': 'para_pr', 'combined':'combined_pr', 'combined_full':'combined_pr_full'}, inplace=True)
            helpers.display_first(df_pr_grouped)
            return df_pr_grouped

        def aggregate_to_article_level_for_pm(df_pm):
            df_pm_grouped = df_pm.groupby('pmid').apply(my_aggr).reset_index()
            df_pm_grouped.rename(columns={'paragraph': 'para_pm', 'combined':'combined_pm', 'combined_full':'combined_pm_full'}, inplace=True)
            helpers.display_first(df_pm_grouped)
            return df_pm_grouped

        def add_column_to_eureka(df_pr, feature='ea_source'):
            if feature == 'ea_source':
                df_ea_source = pd.read_csv(f'{self.data_folder}/{DATA_FILE_TO_WORK_EUREKA}', usecols='eaid contact institution'.split())
                df_ea_source.fillna('', inplace=True)
                df_ea_source['ea_source'] = df_ea_source.apply(helpers.get_type_of_eureka_source, axis=1)
                print(df_ea_source.ea_source.value_counts())
                helpers.display_first(df_ea_source)
                helpers.display_first(df_pr)
                df_pr = df_pr.merge(df_ea_source, how='left')
                helpers.display_first(df_pr)
                #helpers.display_df_cnt(df, 'year', n=20)
                return df_pr
            else:
                print(f'- error: not implemented for: "{feature}"')
                sys.exit()

        nrows = None # takes 20 seconds
        #nrows = 3000
        df_pm, df_pr = self._load_sentence_level_claim_strength_prediction_data_for_both_pubmed_and_eureka(nrows)

        # merge df_pm and df_pr based on their pmid
        print('\n- the following procedure may take 20 to 30 seconds')
        df_pm_grouped = aggregate_to_article_level_for_pm(df_pm)
        df_pr_grouped = aggregate_to_article_level_for_pr(df_pr)

        df = df_pr_grouped.merge(df_pm_grouped, on='pmid')
        df = add_column_to_eureka(df, 'ea_source')
        helpers.display_first(df)

        df = df.rename(columns={
            'combined_pr' : 'ea_pred',
            'combined_pr_full' : 'ea_pred_detail',
            'para_pr' : 'ea_sentences',
            'combined_pm' : 'pm_pred',
            'combined_pm_full' : 'pm_pred_detail',
            'para_pm' : 'pm_sentences'
        })

        df['eaid'] = df['eaid'].apply(lambda x: f'{x}.php')
        cols = 'eaid pmid year ea_source ea_pred pm_pred ea_pred_detail pm_pred_detail ea_sentences pm_sentences'.split()
        fpath_out = self._get_fpath_of_aggregated_pm_and_pr()
        df[cols].to_csv(fpath_out, index=False)
        print('- output:', fpath_out)


    def save_contact_domain_name_frequency_for_manually_checking_sources_of_eureka_press_releases(self):
        df = pd.read_csv(f'{self.data_folder}/{DATA_FILE_TO_WORK_EUREKA}', usecols='eaid contact institution'.split())
        df.fillna('', inplace=True)
        dt = df.groupby('contact').agg(freq=('contact', 'count'),
                institutions = ('institution', lambda x: '|'.join([e for e in set(list(x)) if e])))\
                .reset_index()
        dt = dt[dt.contact.str.find('edu')<0]
        dt = dt[dt.contact.str.find('ac.')<0]
        dt = dt[~dt.contact.isin(helpers.KNOWN_JOURNAL_LIST)]
        dt = dt.sort_values('freq', ascending=False)
        fpath_out = f'{self.working_folder}/contact_domain_name_freq.csv'
        dt.to_csv(fpath_out, index=False)
        print('- out:', fpath_out)


    '''
    def _load_pm_pr_exagg_data_with_pr_institution(self):
        df_ea_source = pd.read_csv(f'{self.data_folder}/21k_eureka_beginning_sentences.csv', usecols='eaid contact institution'.split())
        df_ea_source.fillna('', inplace=True)
        df_ea_source['ea_source'] = df_ea_source.apply(self.get_type_of_eureka_source, axis=1)
        print(df_ea_source.ea_source.value_counts())
        print(df_ea_source.head())

        df = pd.read_csv(self._get_fpath_of_aggregated_pm_and_pr())
        df['eaid'] = df['eaid'].apply(lambda x: f'{x}.php')
        helpers.display_df_cnt(df, 'year', n=20)
        df = df.merge(df_ea_source, how='left')
        helpers.display_df_cnt(df, 'year', n=20)
        return df
    '''


    def load_preprocessed_exagg_data_that_are_observational_study(self, pm_claim_type_wanted=['correlational']):
        fpath_obs_pred = f'{self.working_folder}/{TAG}_{PREDICTED_OBSERVATIONAL_FILE_PUBMED}'
        df_obs = pd.read_csv(fpath_obs_pred)
        print('\n- pubmed papers before applying observational study classifier:', len(df_obs))
        df_obs = df_obs[df_obs.pred_proba >= self.observational_study_prediction_confidence_TH]
        print('- pubmed papers after applying observational study classifier:', len(df_obs))

        df = pd.read_csv(self._get_fpath_of_aggregated_pm_and_pr())

        df = df[df.pmid.isin(df_obs.pmid)]
        #print(df.year.value_counts())
        print('\n- press releases whose corresponding paper are predicted as observational studies:', len(df))

        if pm_claim_type_wanted != 'all':
            df = df[df.pm_pred.isin(pm_claim_type_wanted)]
        print(f'\n- press releases with corresponding observational study being predicted within {pm_claim_type_wanted}: {len(df)}')

        #if len(pm_claim_type_wanted)==1 and pm_claim_type_wanted[0] == 'correlational':
        df['ea_is_causal'] = df.ea_pred.apply(lambda x: int(x=='causal'))
        return df

    def plot_trend_of_exaggeration(self):
        combined_approach = self.combined_approach
        headline_or_three = self.headline_or_three

        if self.use_causal2causal_exaggeration:
            pm_claim_type_wanted = ['causal']
        else:
            pm_claim_type_wanted = ['correlational']
        print(f'\n>> {pm_claim_type_wanted[0]}-to-causation exaggeration\n')
        df = self.load_preprocessed_exagg_data_that_are_observational_study(pm_claim_type_wanted)

        print(f'\n- Overall exaggerations: {df.ea_is_causal.sum()} / {len(df)} = {df.ea_is_causal.mean():.3f}')

        df_by_year = df.groupby('year')['ea_is_causal'].aggregate(rate='mean', count='count').reset_index()
        df_by_year['count_exagg'] = (df_by_year['count'] * df_by_year.rate).astype(int)
        print(df_by_year)

        tmp = df_by_year[df_by_year.year>=2010]
        corr, pv = stats.spearmanr(tmp.year, tmp.rate)
        print(f'\nSpearman correlation coef: {corr:.3f}   pvalue: {pv:.5f}\n')

        if self.use_causal2causal_exaggeration:
            ylim = (.4, .7)
            ylabel1 = "Causation-to-causation rate"
            ylabel2 = 'Cases of causation-to-causation'
            fig_affix = 'c2c'
            color = 'tab:red'
        else:
            ylim = (0.1, .4)
            ylim = (0.12, .32)
            ylabel1 = "Exaggeration rate"
            ylabel2 = 'Cases of correlation-to-causation'
            fig_affix = 'exagg'
            color = 'tab:blue'

        sns.set(font_scale = 1.3)
        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        ax2 = ax1.twinx()
        ax2.grid(False)

        dy = df_by_year[df_by_year.count_exagg>=10]
        ax2.bar(dy.year, dy['count_exagg'], width=.7, alpha=0.4)
        print('- bars:', len(dy))
        for year, cnt in zip(dy.year, dy.count_exagg):
            fs = 12 - (len(dy)-15)//3
            ax2.annotate(str(cnt), xy=(year, cnt), ha='center', va='bottom', fontsize=fs)

        ax1.plot(dy.year, dy.rate, marker=None, ls='-', label='exaggeration', lw=4, color=color)

        if headline_or_three=='headline':
            ax1.set_ylim((0.05, 0.65))
            ax1.legend(loc='upper left', prop={'size': 12}, facecolor='white', framealpha=1)
        else:
            ax1.set_ylim(ylim)

        ax1.set_ylabel(ylabel1)
        ax2.set_ylabel(ylabel2)
        self.save_figure(fig, f'_trend_{combined_approach}_{headline_or_three}_{fig_affix}.pdf')


    def plot_university_vs_journal(self):
        df = self.load_preprocessed_exagg_data_that_are_observational_study()

        df = df[df.ea_source.isin(['University', 'Journal'])].copy()
        print('- press releases whose ea_source is university or journal:', len(df))
        df = df.sort_values('ea_source', ascending=False)

        dg = df.groupby('ea_source').agg(cnt=('ea_is_causal', 'count'), exagg_ratio = ('ea_is_causal', 'mean')).reset_index()
        print(dg)
        ac_ratio = dg[dg.ea_source=="University"].iloc[0].exagg_ratio
        jo_ratio = dg[dg.ea_source=="Journal"].iloc[0].exagg_ratio
        print(f'\n- university vs journal ratio: { ac_ratio/jo_ratio :.3f}\n')

        sns.set(font_scale = 1.8)
        plt.figure(figsize = (5.0, 4.0))

        hue_name = 'Source of press release'
        df = df.rename(columns={'ea_source': hue_name})
        ax = sns.barplot(x=hue_name, y="ea_is_causal", data=df)
        plt.ylabel('Exaggeration rate')
        ax.set_xlabel('')

        filename = f'_barplot_university_vs_journal_{self.combined_approach}_{self.headline_or_three}.pdf'
        self.save_figure(ax.figure, filename, tight=1)


    def plot_venn_diagram(self):
        from matplotlib_venn import venn3

        df = self.load_preprocessed_exagg_data_that_are_observational_study('all')
        helpers.display_first(df)

        def get_claim_types(preds):
            out = []
            for claim_proba in preds.split(' // '):
                claim, proba = claim_proba.split()
                if claim[0] == 'c' and float(proba)>=self.claim_confidence_TH:
                    out.append(claim)
            return ' '.join(sorted(list(set(out))))


        df['claim_types'] = df.pm_pred_detail.apply(get_claim_types)
        print(df['claim_types'][:10])
        dg = df.groupby('claim_types').agg(cnt=('claim_types','count')).reset_index()
        dg['ratio'] = dg.cnt/dg.cnt.sum()
        print(f'\n- before removing the NA conclusion: {dg.cnt.sum():,}\n')
        print(dg)
        dg = df[df.claim_types!=''].groupby('claim_types').agg(cnt=('claim_types','count')).reset_index()
        dg['ratio'] = dg.cnt/dg.cnt.sum()
        print(f'\n- after removing the NONE conclusion: {dg.cnt.sum():,}\n')
        print(dg)

        names =   'Abc aBc ABc abC AbC aBC ABC'.split()
        mapping = {'A':'correlational', 'B':'causal', 'C':'conditional'}
        percentages, counts = [], []
        for name in names:
            claim_types = ' '.join(sorted([mapping[e] for e in name if e in mapping]))
            ratio = dg[dg.claim_types==claim_types].iloc[0].ratio
            cnt = dg[dg.claim_types==claim_types].iloc[0].cnt
            percentage = float(f'{ratio*100:.1f}')
            percentages.append(percentage)
            counts.append(cnt)
            print(name, claim_types, cnt, percentage)

        venn = venn3(subsets = percentages, set_labels = (r'Correlational', 'Direct causal', 'Conditional causal') )
        for text in venn.set_labels:
            text.set_fontsize(20)
        for text in venn.subset_labels:
            text.set_fontsize(18)

        ids = ['100', '010', '110', '001', '101', '011', '111']
        for id, percentage, count in zip(ids, percentages, counts):
            text = str(count) + "\n" + r'\textrm{' + str(percentage) + r'\%}'
            venn.get_label_by_id(id).set_text(text)

        # set position of the title of each circle
        lbl = venn.get_label_by_id("A") # correlational
        x, y = lbl.get_position()
        lbl.set_position((x+0.15, y-0.10))
        lbl = venn.get_label_by_id("B") # causal
        x, y = lbl.get_position()
        lbl.set_position((x-0.38, y+0.03))
        lbl = venn.get_label_by_id("C") # conditional
        x, y = lbl.get_position()
        lbl.set_position((x-0.45, y+0.15))

        fig_out = '_venn.pdf'
        self.save_figure(plt, fig_out, tight=1)


    def export_some_cases_to_html_for_further_study(self):
        df = self.load_preprocessed_exagg_data_that_are_observational_study(pm_claim_type_wanted=['causal'])
        print('\n- eureka cases when pm_pred = causal\n')
        print(df.ea_pred.value_counts())

        for ea_pred in ('causal', 'non_causal', 'na'):
            ds = df[df.ea_pred==ea_pred].sample(30, random_state=0)
            print(ea_pred, len(df), len(ds))
            cnt = 0
            html = ["<style>body{margin:auto;width:900px} *{font-family:arial; line-height:25px} div{margin:10px; background:#eee; padding:10px} a{text-decoration:none}</style><body>"]
            html.append(f'<h1> paper causal => eureka {ea_pred}</h1>')
            for _, d in ds.iterrows():
                cnt += 1
                ea_sentences = d.ea_sentences.split(' // ')
                ea_headline = f'{ea_sentences[0]} (<font color=gray>{d.ea_pred}</font>)'

                item = f'''<h3>{cnt}. <a target=_blank href=https://www.eurekalert.org/pub_releases/{d.eaid}>{ea_headline}</a></h3>
<div> <b>[PAPER]</b> {d.pm_sentences}<p> {d.pm_pred_detail}  (<b>aggregated</b>: {d.pm_pred}) </div>
<div> <b>[EUREKA]</b> {d.ea_sentences}<p> {d.ea_pred_detail} </div>
'''
                html.append(item.replace(' // ', ' <font color=red>//</font>  '))

            HTML_FOLDER = config.get('analysis', 'HTML_FOLDER')
            fpath_out = f'{HTML_FOLDER}/sample_of_pm_causal__vs__{ea_pred}.html'
            with open(fpath_out, 'w') as fout:
                fout.write('\n'.join(html))
                print('- out:', fpath_out)


    def plot_number_of_observational_studies_over_years(self):
        df = self.load_preprocessed_exagg_data_that_are_observational_study('all')
        print(df.pm_pred.value_counts())
        print()

        df_list = [df]
        df_list.append( df[df.pm_pred!='na'] )
        df_list.append( df[df.pm_pred=='correlational'] )
        df_list.append( df[df.pm_pred=='causal'] )

        labels = ['All observational studies',
                'All observational studies' + '\n' + r'with a \textbf{unanimous} claim',
                'Observational studies with' + '\n' + r'\textbf{correlational} findings',
                'Observational studies with' + '\n' + r'\textbf{causal} findings' ]
        extra_indexes_of_causal2causal = (1,3)
        dff = None
        for i, df_ in enumerate(df_list):
            if not self.use_causal2causal_exaggeration:
                if i in extra_indexes_of_causal2causal:
                    continue
            dg = df_.groupby('year')['eaid'].aggregate(count='count').reset_index()
            dg['type'] = labels[i] + '\n' + f'({len(df_):,} press releases)'
            print(dg)
            dff = dg if dff is None else pd.concat((dff, dg))

        print()
        print(dff['type'].value_counts())

        sns.set(font_scale = 1.3)
        figsize = (6.5, 4)
        plt.subplots(figsize=figsize)
        if self.use_causal2causal_exaggeration:
            palette = ['tab:gray', 'tab:green', 'tab:blue', 'tab:red']
        else:
            palette = ['tab:gray', 'tab:blue']
        ax = sns.lineplot(x='year', y='count', hue='type', palette=palette, marker=None, data=dff, lw=4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], fontsize=12)
        plt.xlabel('')
        plt.ylabel('Number of press releases')
        fig_out = '_studies_vs_year.pdf'
        self.save_figure(ax.figure, fig_out, tight=0)


    def _get_fig_fpath(self, fname):
        #import socket
        #if socket.gethostname().lower().find('macbook')>=0:
        folder = config.get('analysis', "EXPORT_FIGURE_FOLDER")
        helpers.get_or_create_dir(folder)
        return f'{folder}/__{fname}'

    def save_figure(self, fig, filename, tight=False):
        if tight:
            fig.tight_layout()
        outfile = self._get_fig_fpath(filename)
        fig.savefig(outfile)
        print('- figure saved:', outfile)
        plt.close()


def main():
    exagg = ExaggerationAnalysis()

    #exagg.aggregate_from_sentence_to_article_level__and__merge_aggregated_pubmed_and_eureka()

    #exagg.plot_number_of_observational_studies_over_years()
    exagg.plot_trend_of_exaggeration()
    #exagg.plot_university_vs_journal()

if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')
