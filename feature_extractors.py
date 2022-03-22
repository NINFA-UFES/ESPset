from adabelief_pytorch import AdaBelief
from tempfile import mkdtemp
from pytorch_metric_learning import losses
from datahandler import BalancedDataLoader
from networks import myEmbeddingNet, NeuralNetTransformer
from skorch.dataset import ValidSplit


def split_gridsearchparams(params):
    grid_search_params = {p: v for p, v in params.items() if isinstance(v, list)}
    params = {p: v for p, v in params.items() if not isinstance(v, list)}

    return params, grid_search_params


def _processLossFunctionParams(loss_function):
    if(isinstance(loss_function, str)):
        if(loss_function == 'tripletloss'):
            loss_function = losses.TripletMarginLoss
        elif(loss_function == 'AngularLoss'):
            loss_function = losses.AngularLoss
        elif(loss_function == 'CosFaceLoss'):
            loss_function = losses.CosFaceLoss
        elif(loss_function == 'ContrastiveLoss'):
            loss_function = losses.ContrastiveLoss
        elif(loss_function == 'GeneralizedLiftedStructureLoss'):
            loss_function = losses.GeneralizedLiftedStructureLoss
        elif(loss_function == 'MarginLoss'):
            loss_function = losses.MarginLoss()
        elif(loss_function == 'NormalizedSoftmaxLoss'):
            loss_function = losses.NormalizedSoftmaxLoss
        elif(loss_function == 'NTXentLoss'):
            loss_function = losses.NTXentLoss
        elif(loss_function == 'ProxyAnchorLoss'):
            loss_function = losses.ProxyAnchorLoss
        elif(loss_function == 'TupletMarginLoss'):
            loss_function = losses.TupletMarginLoss()
        else:
            raise Exception("loss function not recognized! %s" % loss_function)

    return loss_function


def createConvNet(n_classes, last_layer_size, **feature_extractor_params):
    from networks import MyNeuralNetClassifier
    """
    Common neural net classifier.
    """

    if('learning_rate' in feature_extractor_params):
        feature_extractor_params['optimizer__lr'] = feature_extractor_params['learning_rate']
        del feature_extractor_params['learning_rate']

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False,
                            'print_change_log': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    parameters = {
        'device': 'cuda',
        'max_epochs': 300,
        'train_split': ValidSplit(9, stratified=True),
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'iterator_valid': BalancedDataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False}
    parameters = {**parameters, **optimizer_parameters}

    feature_extractor_params, gridsearch_params = split_gridsearchparams(feature_extractor_params)
    parameters.update(feature_extractor_params)

    convnet = MyNeuralNetClassifier(module__n_classes=n_classes,
                                    module__embedding_net=myEmbeddingNet(last_layer_size),
                                    cache_dir=mkdtemp(),
                                    **parameters)

    return convnet, gridsearch_params


def create_MetricLearningNetwork(loss_function='tripletloss', **feature_extractor_params):
    if('learning_rate' in feature_extractor_params):
        feature_extractor_params['optimizer__lr'] = feature_extractor_params['learning_rate']
        del feature_extractor_params['learning_rate']

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False,
                            'print_change_log': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    parameters = {
        'device': 'cuda',
        'module': myEmbeddingNet, 'module__num_outputs': 8,
        'max_epochs': 300,
        'train_split': ValidSplit(9, stratified=True),
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'iterator_valid': BalancedDataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False}
    parameters = {**parameters, **optimizer_parameters}

    ### Renaming "loss_function__" parameters to "criterion__" ###
    loss_func_params = {k.split('__', 1)[1]: v for k, v in feature_extractor_params.items()
                        if 'loss_function__' in k}
    for k in loss_func_params.keys():
        del feature_extractor_params['loss_function__'+k]
    loss_func_params = {"criterion__%s" % k: v for k, v in loss_func_params.items()}
    feature_extractor_params, gridsearch_params = split_gridsearchparams(feature_extractor_params)
    ###############################################################

    parameters.update(feature_extractor_params)
    loss_function_cls = _processLossFunctionParams(loss_function)

    tripletnet = NeuralNetTransformer(**parameters, cache_dir=mkdtemp(),
                                      criterion=loss_function_cls, **loss_func_params)

    return tripletnet, gridsearch_params
