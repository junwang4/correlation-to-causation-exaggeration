import time, sys
import inspect
import fire

import sentence_claim_strength_classification as claim_clf
import observational_study_classification as obs_clf
import exaggeration_analysis

class Client:

    def sentence_claim_strength_classification(self, data_type="pubmed", data_augmentation=False, task="evaluate_and_error_analysis"):
        sig = inspect.signature(self.sentence_claim_strength_classification)
        for param in sig.parameters.values(): print(f'{param.name:15s} = {eval(param.name)}')

        obj = claim_clf.SentenceClassifier(data_type=data_type, data_augmentation=data_augmentation)
        try:
            func = getattr(obj, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()

    def observational_study_classification(self, task=None):
        sig = inspect.signature(self.observational_study_classification)
        for param in sig.parameters.values(): print(f'{param.name:15s} = {eval(param.name)}')

        obj = obs_clf.ObservationalStudyClassifier()
        try:
            func = getattr(obj, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()

    def exaggeration_analysis(self, task=None):
        sig = inspect.signature(self.exaggeration_analysis)
        for param in sig.parameters.values(): print(f'{param.name:15s} = {eval(param.name)}')

        obj = exaggeration_analysis.ExaggerationAnalysis()
        try:
            func = getattr(obj, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()


if __name__ == "__main__":
    tic = time.time()
    fire.Fire(Client)
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')
