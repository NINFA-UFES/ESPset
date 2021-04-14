import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler



class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        if(torch.is_tensor(labels)):
            labels = labels.numpy()
        if(isinstance(labels, list)):
            labels = np.array(labels)
        self.labels_set = list(set(labels))
        self.label_to_indices = {label: np.where(labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class BalancedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0, collate_fn=None,
                 pin_memory=False, worker_init_fn=None):
        targets = BalancedDataLoader.getTargets(dataset)
        if(targets is None):
            super().__init__(dataset, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        else:
            if(batch_size > len(targets)):
                batch_size = len(targets)
            if(torch.is_tensor(targets)):
                targets = targets.cpu().numpy()
            nclasses = len(set(targets))
            sampler = BalancedBatchSampler(targets, nclasses, batch_size//nclasses)
            super().__init__(dataset, num_workers=num_workers, batch_sampler=sampler,
                             collate_fn=collate_fn, pin_memory=pin_memory, worker_init_fn=worker_init_fn)

    @staticmethod
    def getTargets(dataset):
        if(hasattr(dataset, 'y')):
            return dataset.y
        elif(hasattr(dataset, 'targets')):
            return dataset.targets
        if(isinstance(dataset, torch.utils.data.Subset)):
            targets = BalancedDataLoader.getTargets(dataset.dataset)
            return targets[dataset.indices]
        return None
