import torch
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_validate
from GridSearchCV_norefit import GridSearchCV_norefit
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import itertools
from feature_extractors import extract_handcrafted_features

DEFAULT_SCORER = 'f1_macro'
RANDOM_STATE = None

# Uncomment these two lines if you need to ensure exact same results at multiple executions.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def combineTransformerClassifier(transformers, base_classifiers):
    def buildGridSearch(clf, transf_param_grid, base_classif_param_grid):
        """
        A single Grid Search is built for the complete classifier model (ex: tripletnetwork + knn).
        """
        transf_param_grid = {"transformer__%s" % k: v
                             for k, v in transf_param_grid.items()}
        base_classif_param_grid = {"base_classifier__%s" % k: v
                                   for k, v in base_classif_param_grid.items()}
        param_grid = {**transf_param_grid, **base_classif_param_grid}
        return createGridSearch(clf, param_grid, gridsearch_constructor=GridSearchCV_norefit)

    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        classifier = Pipeline([('transformer', transf),
                               ('base_classifier', base_classif)])
        classifier = buildGridSearch(classifier,
                                     transf_param_grid, base_classif_param_grid)

        final_name = '%s + %s' % (transf_name, base_classif_name)
        yield (final_name, classifier)


def createGridSearch(clf, param_grid, n_jobs=None, gridsearch_constructor=GridSearchCV):
    has_gridsearch = np.any([isinstance(v, list) for v in param_grid.values()])
    if (has_gridsearch):
        gridsearch_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        param_grid = {p: v if isinstance(v, list) else [v] for p, v in param_grid.items()}
        return gridsearch_constructor(clf, param_grid, scoring=DEFAULT_SCORER, cv=gridsearch_sampler)
    return clf


def main(signals, features, Y, esp_ids, config):
    X = np.expand_dims(signals[:, :6100], axis=1)  # raw data for deep neural networks.
    base_classifiers = config.getBaseClassifiers()
    sampler = config.getCrossValidator()
    scoring = DEFAULT_SCORER

    Results = {}

    print("Training...")
    fe_list = config.getFeatureExtractors()
    for classifier_name, classifier in combineTransformerClassifier(fe_list, base_classifiers):
        print(classifier_name)
        Results[classifier_name] = cross_validate(classifier, X, Y, groups=esp_ids, scoring=scoring, cv=sampler)

    if (config.train_single_classifiers):
        df_features = extract_handcrafted_features(signals, starting_idx_pos=100)
        X = df_features.values.astype(np.float32)
        for classif_name, classifier, param_grid in base_classifiers:
            print(classif_name)
            classifier = createGridSearch(classifier, param_grid, n_jobs=-1)
            scores = cross_validate(classifier, X, Y, groups=esp_ids, scoring=scoring, cv=sampler)
            Results[classif_name] = scores

    ## Save results##
    if (config.save_file is not None):
        results_asmatrix = []
        for classif_name, result in Results.items():
            print("===%s===" % classif_name)
            for rname, rs in result.items():
                if (rname.startswith('test_') or 'time' in rname):
                    if (rname.startswith('test_')):
                        metric_name = rname.split('_', 1)[-1]
                    else:
                        metric_name = rname
                    print("%s: %f" % (metric_name, rs.mean()))
                    for i, r in enumerate(rs):
                        results_asmatrix.append((classif_name, metric_name, i+1, r))
        df = pd.DataFrame(results_asmatrix, columns=['classifier name', 'metric name', 'fold id', 'value'])
        df.to_csv(config.save_file, index=False)


if __name__ == '__main__':
    import argparse
    from yaml_loader import loadConfiguration

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('-i', '--inputdata', type=str, default='data')
    parser.add_argument('-o', '--outfile', type=str, required=False)
    args = parser.parse_args()

    config = loadConfiguration(args.config, args.inputdata, args.outfile)

    RANDOM_STATE = config.random_seed
    if (RANDOM_STATE is not None):
        np.random.seed(RANDOM_STATE)
        torch.cuda.manual_seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

    print("Loading data...")
    signals = np.loadtxt('%s/spectrum.csv' % args.inputdata, delimiter=';', dtype=np.float32)
    signals = signals[:, 100:]  # The first 100 data points are usually just noise.
    features = pd.read_csv('%s/features.csv' % args.inputdata, sep=';', index_col='id')
    labels, _ = features['label'].factorize()
    if ('esp_id' in features):
        esp_id = features['esp_id']
    else:
        esp_id = None

    main(signals, features, labels, esp_id, config)
