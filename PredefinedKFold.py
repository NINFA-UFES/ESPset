import numpy as np
from sklearn.model_selection._split import _BaseKFold

class PredefinedKFold(_BaseKFold):
    def __init__(self, n_rounds='all'):
        self.folds_gids = [[[0, 2, 5, 7, 9], [1, 6], [3, 4], [8, 10]],
                           [[0, 2, 3, 4], [1, 6], [5], [7, 8, 9, 10]],
                           [[0, 2], [1, 6, 8, 10], [3, 4], [5, 7, 9]],
                           [[0, 1, 3, 8, 9], [4, 10], [5, 7], [2, 6]],
                           [[0, 2, 3, 7, 8, 9], [1], [4, 10], [5, 6]],
                           [[0, 1], [5, 7], [2, 3, 4, 6, 9], [8, 10]],
                           [[0, 3, 6, 8, 10], [1], [2, 5, 7], [4, 9]],
                           [[0, 8, 10], [3, 6], [5, 7], [1, 2, 4, 9]]]
        if(n_rounds != 'all'):
            self.folds_gids = self.folds_gids[:n_rounds]
        self.n_splits = self.get_n_splits()

    def _iter_test_indices(self, X, y, groups):
        groups = np.array(groups)
        idxs_special, = np.where(y == 3)  # desalinhamento
        for fold_round in self.folds_gids:
            k = 0
            for i, fgids in enumerate(fold_round):
                test_idxs, = np.where((np.isin(groups, fgids)) & (y != 3))
                if(i < len(fold_round)-1):
                    to_add = round(len(test_idxs)/len(y) * len(idxs_special))
                    to_add = max(1, to_add)
                    test_idxs = np.append(test_idxs, idxs_special[k:k+to_add])
                    k += to_add
                else:
                    test_idxs = np.append(test_idxs, idxs_special[k:])

                yield test_idxs

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds_gids) * len(self.folds_gids[0])

    def get_validation_fold_idxs(self):
        n = len(self.folds_gids)
        nfolds = len(self.folds_gids[0])
        return _get_validation_fold_idxs(n, nfolds)
