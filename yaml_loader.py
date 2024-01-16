import sklearn
from sklearn.model_selection import GridSearchCV
from PredefinedKFold import PredefinedKFold
from pathlib import Path
import yaml
import importlib
import numpy as np

_attrs = [
    'train_single_classifiers',
    'feature_extractors',
    'cross_validation',
    'random_seed',
    'base_classifiers'
]

YAML_INVALID_ERROR_MSG =\
    """YAML invalid parameters: %s
    Valid parameters: %s
    """


def str2class(class_fullname: str):
    spl = class_fullname.split('.')
    module_name = '.'.join(spl[:-1])
    class_name = spl[-1]
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


class Config:
    cross_validators = {'stratifiedkfold': sklearn.model_selection.StratifiedKFold,
                        'StratifiedShuffleSplit'.lower(): sklearn.model_selection.StratifiedShuffleSplit,
                        'RepeatedStratifiedKFold'.lower(): sklearn.model_selection.RepeatedStratifiedKFold,
                        'predefinedkfold': PredefinedKFold}

    param_mapping = {'sklearn.neighbors.KNeighborsClassifier': {'k': 'n_neighbors'}}

    @staticmethod
    def mapClassifierName(clf_name):
        name = clf_name.lower()
        if(name in ['KNeareastNeighbors'.lower(), 'knn']):
            return 'sklearn.neighbors.KNeighborsClassifier'
        if(name in ['RandomForest'.lower()]):
            return 'sklearn.ensemble.RandomForestClassifier'
        return clf_name

    def __init__(self, dataset_path, save_file, **kwargs):
        self.train_single_classifiers: bool
        self.feature_extractors: dict
        self.cross_validation: dict
        self.random_seed: int

        # Check invalid keys
        attr_set = set(_attrs)
        kwkeyset = set(kwargs.keys())
        if(len(kwkeyset - attr_set) > 0):
            raise RuntimeError(YAML_INVALID_ERROR_MSG % (kwkeyset-attr_set, set(_attrs)))
        # if not ((attr_set & kwkeyset) == attr_set):
        #     raise RuntimeError("YAML key error.")

        for key in _attrs:
            if(key in kwargs):
                value = kwargs[key]
                setattr(self, key, value)

        self.dataset_path = dataset_path
        self.save_file = save_file

    def __repr__(self):
        repr_ = ""
        for i in _attrs:
            repr_ += "%s: %s\n" % (i, repr(getattr(self, i)))
        return repr_

    def getCrossValidator(self) -> sklearn.model_selection.BaseCrossValidator:
        sampler_config = self.cross_validation
        sampler_class = Config.cross_validators[sampler_config['class'].lower()]
        del sampler_config['class']
        return sampler_class(**sampler_config)

    @staticmethod
    def split_gridsearchparams(params):
        grid_search_params = {p: v for p, v in params.items() if isinstance(v, list)}
        params = {p: v for p, v in params.items() if not isinstance(v, list)}

        return params, grid_search_params

    def getBaseClassifiers(self):
        def parameters_abbreviation(clf_name, p):
            if(clf_name in Config.param_mapping):
                params = Config.param_mapping[clf_name]
                if(p in params):
                    return params[p]
            return p

        def createScaler(scaler_params):
            if(scaler_params.strip().lower() in ['standardscaler', 'standard scaler']):
                return sklearn.preprocessing.StandardScaler()
            return None

        base_classifiers = []
        for bclf in self.base_classifiers:
            assert(len(bclf) == 1)
            clf_orig_name, clf_params = next(iter(bclf.items()))
            clf_name = Config.mapClassifierName(clf_orig_name)
            clf_class = str2class(clf_name)

            ###Processing classifier parameters###
            clf_params = {parameters_abbreviation(clf_name, p): v for p, v in clf_params.items()}
            if('name' in clf_params):
                name = clf_params['name']
                del clf_params['name']
            else:
                name = clf_orig_name

            if('scaler' in clf_params):
                scaler = createScaler(clf_params['scaler'])
                del clf_params['scaler']
            else:
                scaler = None

            clf_params, gridsearch_params = Config.split_gridsearchparams(clf_params)
            #####################################
            clf = clf_class(**clf_params)

            if(scaler is not None):
                clf = sklearn.pipeline.Pipeline([('scaler', scaler),
                                                 ('clf', clf)])
                gridsearch_params = {'clf__%s' % p: v for p, v in gridsearch_params.items()}

            base_classifiers.append((name, clf, gridsearch_params))

        return base_classifiers

    def getFeatureExtractors(self):
        if(not hasattr(self, 'feature_extractors')):
            return []
        fe_list = []
        for fe in self.feature_extractors:
            assert(len(fe) == 1)
            fe_name, fe_params = next(iter(fe.items()))
            fe_class = str2class(fe_name)
            if('name' in fe_params):
                fe_name = fe_params['name']
                del fe_params['name']

            fe, fe_params = fe_class(**fe_params)
            fe_list.append((fe_name, fe, fe_params))

        return fe_list


def loadConfiguration(yaml_file, dataset_path, save_file):
    if isinstance(yaml_file, str):
        file = Path(yaml_file)

    with open(yaml_file) as yaml_stream:
        f = yaml.load(yaml_stream, Loader=yaml.FullLoader)
    return Config(dataset_path, save_file, **f)

