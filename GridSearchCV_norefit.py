from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import indexable, _check_fit_params
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier
from sklearn.metrics._scorer import _check_multimetric_scoring
from itertools import product
import numpy as np



class GridSearchCV_norefit(GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, refit=False, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(estimator, param_grid, scoring=scoring,
                         n_jobs=n_jobs, refit=False, cv=cv,
                         verbose=verbose, pre_dispatch=pre_dispatch,
                         error_score=error_score, return_train_score=return_train_score)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose,
                                    return_estimator=True)  # added
        results = {}
        best_estimator = None
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params, cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                best_index = np.argmax([s['score'] for s in list(zip(*out))[0]])
                nonlocal best_estimator
                best_estimator = out[best_index][-1]
                for o in out:
                    o.pop()

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        self.best_estimator_ = best_estimator
        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def transform(self, X):
        return self.best_estimator_.transform(X)
