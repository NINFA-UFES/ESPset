from datetime import datetime
import os
from callbacks import LoadEndState
from tempfile import mkdtemp
from torch.nn import functional as F
from skorch.utils import to_tensor, to_device
from skorch.dataset import unpack_data
from torch.utils.tensorboard import SummaryWriter
from skorch import NeuralNet
from sklearn.base import TransformerMixin
import numpy as np
from datahandler import BalancedDataLoader
import torch
from torch import nn
import skorch
from adabelief_pytorch import AdaBelief

CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')


class MLP_module(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        hidden_layer_size = 2*num_inputs+1
        self.fc = nn.Sequential(nn.Linear(num_inputs, hidden_layer_size), nn.LeakyReLU(negative_slope=0.05),
                                nn.Linear(hidden_layer_size, hidden_layer_size), nn.LeakyReLU(negative_slope=0.05),
                                nn.Linear(hidden_layer_size, num_outputs),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, x):
        return self.fc(x)


class Softmax_module(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(num_inputs, num_outputs), nn.LeakyReLU(negative_slope=0.05),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, x):
        return self.fc(x)


class TensorBoardCallback(skorch.callbacks.TensorBoard):
    def on_train_end(self, net, X=None, y=None, **kwargs):
        neural_net_params = {k: v for k, v in net.get_params().items() if type(v) in (int, float, str, bool)}
        neural_net_params["NeuralNet class"] = net.__class__.__name__
        if('valid_loss' in net.history[-1]):
            loss_name = 'valid_loss'
        else:
            loss_name = 'train_loss'
        best_loss = min(net.history[:, loss_name])
        self.writer.add_hparams(neural_net_params, {'hparam/best_loss': best_loss})


class NeuralNetBase(NeuralNet):
    def __init__(self, module, *args, cache_dir=mkdtemp(), init_random_state=None, validation_dataset=None, monitor_loss='valid_loss', **kwargs):
        super().__init__(module, *args, **kwargs)
        self.validation_dataset = validation_dataset
        self.init_random_state = init_random_state
        self.monitor_loss = monitor_loss
        self.cache_dir = cache_dir

    def get_default_callbacks(self):
        import socket
        from pathlib import Path

        checkpoint_callback = skorch.callbacks.Checkpoint(dirname=self.cache_dir,
                                                          monitor=self.monitor_loss + '_best',  # monitor='non_zero_triplets_best',  # monitor='train_loss_best'
                                                          f_params='best_epoch_params.pt',
                                                          f_history=None, f_optimizer=None, f_criterion=None)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', CURRENT_TIME, current_time + '_' + socket.gethostname())
        callbacks = [TensorBoardCallback(SummaryWriter(log_dir=log_dir))]
        callbacks += [checkpoint_callback, LoadEndState(checkpoint_callback, delete_checkpoint=True)]

        callbacks.append(skorch.callbacks.EarlyStopping(monitor=self.monitor_loss, patience=100, threshold=1e-4))

        return super().get_default_callbacks()+callbacks

    def fit(self, X, y, **fit_params):
        if(self.cache_dir is not None):
            cache_filename = self.get_cache_filename(X, y)
            if(os.path.isfile(cache_filename)):
                if not self.warm_start or not self.initialized_:
                    self.initialize()
                print("loading cached neuralnet '%s'" % cache_filename)
                self.load_params(f_params=cache_filename)

                return self
            super().fit(X, y, **fit_params)

            # Only save if user did not interrupted the training process.
            if(len(self.history) == self.max_epochs):
                self.save_params(f_params=cache_filename)
            else:
                for _, cb in self.callbacks_:
                    if(isinstance(cb, skorch.callbacks.EarlyStopping)):
                        if(cb.misses_ == cb.patience):
                            self.save_params(f_params=cache_filename)
                            break

        else:
            super().fit(X, y, **fit_params)
        return self

    def initialize(self):
        if(self.init_random_state is not None):
            np.random.seed(self.init_random_state)
            torch.cuda.manual_seed(self.init_random_state)
            torch.manual_seed(self.init_random_state)
        return super().initialize()

    def set_validation_dataset(self, X, y):
        self.validation_dataset = self.get_dataset(X, y)

    def get_split_datasets(self, X, y=None, **fit_params):
        if(self.validation_dataset is None):
            return super().get_split_datasets(X, y, **fit_params)
        dataset_train = self.get_dataset(X, y)
        return dataset_train, self.validation_dataset

    def get_params(self, deep=True, **kwargs):
        params = super().get_params(deep=deep, **kwargs)
        params = {k: v for k, v in params.items() if not('callbacks' in k)}
        return params

    def get_cache_filename(self, X, y) -> str:
        import hashlib

        if(isinstance(X, dict)):
            Xf = X['X']
        else:
            Xf = X

        m = hashlib.md5()
        m.update(self.__class__.__name__.encode('utf-8'))
        n = len(Xf)

        for k, v in self.get_params().items():
            if(k == 'cache_dir'):
                continue
            if(type(v) in (int, float, bool, str)):
                s = '%s:%s' % (str(k), str(v))
            elif(type(v) in (dict, list, tuple)):
                s = '%s:%s' % (str(k), str(len(v)))
            elif(isinstance(v, type)):
                s = '%s:%s' % (str(k), v.__name__)
            else:
                s = '%s:%s' % (str(k), v.__class__.__name__)
            m.update(s.encode('utf-8'))
        m.update(str(n).encode('utf-8'))
        m.update(str(Xf[0]).encode('utf-8'))
        m.update(str(Xf[1]).encode('utf-8'))

        m.update(str(y[0]).encode('utf-8'))
        m.update(str(y.max()).encode('utf-8'))
        m.update(str(y[:n//2].sum()).encode('utf-8'))
        m.update(str(y[n//2:].sum()).encode('utf-8'))

        fname = m.hexdigest()
        return f"{self.cache_dir}/{fname}.pkl"


class NeuralNetTransformer(NeuralNetBase, TransformerMixin):
    def __init__(self, module, init_random_state=None, validation_dataset=None, *args, **kwargs):
        super().__init__(module, *args, init_random_state=init_random_state, validation_dataset=validation_dataset, **kwargs)

    def transform(self, X):
        if(len(X.shape) == 2):
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return self.predict(X)

    def fit(self, X, y=None, **fit_params):
        if(len(X.shape) == 2):
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return super().fit(X, y, **fit_params)


class myEmbeddingNet(nn.Module):
    def __init__(self, num_outputs, num_inputs_channels=1):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(num_inputs_channels, 16, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Flatten()
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94, 192),
                                nn.LeakyReLU(negative_slope=0.05),
                                nn.Linear(192, num_outputs)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = self.fc(output)
        return output

    def encode(self, x):
        with torch.no_grad():
            if(len(x.shape) == 1):
                n = 1
                x = torch.tensor(x[:6100], dtype=torch.float32).cuda()
                x = x.reshape((1, 1, 6100))
                return self.forward(x).squeeze()
            else:
                n = x.shape[0]
                ret = torch.empty((n, self.num_outputs), dtype=torch.float32).cuda()
                k = 0
                for i in range(0, n, 8):
                    batch = torch.tensor(x[i:i+8, :6100], dtype=torch.float32).cuda()
                    batch = batch.reshape(batch.shape[0], 1, 6100)
                    output = self.forward(batch).squeeze()
                    ret[k:k+len(output)] = output
                    k += len(output)
                return ret


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.ReLU()
        for last_module in embedding_net.modules():
            pass
        self.fc1 = nn.Linear(last_module.out_features, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class MyNeuralNetClassifier(skorch.NeuralNetClassifier, NeuralNetBase, TransformerMixin):

    def __init__(self, module__embedding_net, module__n_classes, init_random_state,
                 module=ClassificationNet, **kwargs):
        super().__init__(module=module, module__embedding_net=module__embedding_net, module__n_classes=module__n_classes, **kwargs)
        self.init_random_state = init_random_state
        self.module__embedding_net = module__embedding_net
        self.module__n_classes = module__n_classes

    def transform(self, X):
        with torch.no_grad():
            X = torch.from_numpy(X).cuda()
            X = self.module__embedding_net(X)
            return X.cpu().numpy()


class MLPClassifier(skorch.NeuralNetClassifier, NeuralNetBase):
    def __init__(self, num_inputs, num_outputs, module=MLP_module, device='cuda',
                 train_split=skorch.dataset.CVSplit(0.25, stratified=True, random_state=0),
                 iterator_train=BalancedDataLoader, **kwargs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        module_params = {'num_inputs': num_inputs, 'num_outputs': num_outputs}
        module_params = {'module__'+k: v for k, v in module_params.items()}

        optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                                'eps': 1e-16, 'betas': (0.9, 0.999),
                                'weight_decouple': True, 'rectify': False,
                                'print_change_log': False}  # default
        for k in optimizer_parameters.keys():
            if(k in kwargs):
                optimizer_parameters[k] = kwargs[k]

        optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
        optimizer_parameters['optimizer'] = AdaBelief

        parameters = optimizer_parameters
        parameters.update({'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
                           'iterator_valid': BalancedDataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False})
        parameters.update(module_params)
        parameters.update(kwargs)

        super().__init__(module, device=device, train_split=train_split, iterator_train=iterator_train,
                         **parameters)


class SoftmaxClassifier(MLPClassifier):
    def __init__(self, num_inputs, num_outputs, module=Softmax_module, **kwargs):
        super().__init__(num_inputs, num_outputs, module=module, **kwargs)
