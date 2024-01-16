### Table of Contents  
<!-- toc -->

- [ESPset dataset description](#espset-dataset-description)  
    - [spectrum.csv](#spectrumcsv)  
    - [features.csv](#featurescsv)  
- [Experimental Framework](#experimental-framework)  
    - [Requirements](#requirements)
    - [How to run (.yaml configuration file)](#how-to-run-yaml-configuration-file)
    - [Benchmark classifiers](#benchmark-classifiers)
    - [Paper Experiments](#paper-experiments)

<!-- tocstop -->

# ESPset dataset description
This repository provides the ESPset dataset, a real-world dataset for vibration-based fault diagnosis of electric submersible pumps used on offshore oil exploration.
In addition to that, this repository also provides an experimental framework for adequately comparing research works based on the ESPset dataset and defines some benchmark classifiers.
The ESPset dataset is a collection of vibration signals acquired from accelerometers strategically attached onto the components of Eletrical Submersible Centrifugal Pumps (ESP).
An ESP belong to a class of equipment used in the extraction and exploration of oil and gas subject to severe working conditions.
An ESP system consists of a coupled set of one or more electric motors, pumps and  protectors.

The dataset is provided in two files: spectrum.csv and features.csv in https://drive.google.com/drive/folders/1Tr0ddbopjVgHtccYg5nVcsjMraWkJOaM?usp=sharing

## spectrum.csv
This csv file is a matrix of 6032 lines and 12103 columns, whose values are float numbers seperated by a ';'. 
Each line of this file contains the spectrum of a single vibration signal collected from a sensor at a specific test condition of the ESP.
Each value is the amplitude in inches per second (velocity) at a specific frequency.
Each signal is normalized by the rotation frequency in which the ESP operates, in such a way that the amplitude with respect to the rotation frequency is always at the same position for all signal arrays.

A simple way to load this matrix in python:
```python
import numpy as np
signals = np.loadtxt('data/spectrum.csv', delimiter=';')
```

## features.csv
This csv file of 6033 lines (one line for each signal + a header), contains some features and the labels for all signals:
- **esp_id**: The id of the ESP.
- **label**: The classification label.

Let $X$ be defined as the rotation frequency of the ESP.
Each feature is defined as:
- **median(8,13)**: Median of the amplitudes in the interval (8% $X$, 13% $X$);
- **median(98,102)**: Median of the amplitudes in the interval (98% $X$, 102% $X$);
- **a**: Coefficient $a$ of the exponential regression of type $e^{(a\cdot A+b)}$ where $A$ is an array of equally separated relative frequencies up to $0.4X$, excluding zero. Example: $A=$(0.01, 0.02, ..., 0.39, 0.4).
- **b**: Coefficient $b$ of the exponential regression of type $e^{(a\cdot A+b)}$ where $A$ is an array of equally separated relative frequencies up to $0.4X$, excluding zero. Example: $A=$(0.01, 0.02, ..., 0.39, 0.4).
- **peak1x**: Amplitude in $X$;
- **peak2x**: Amplitude in $2X$;
- **rms(98,102)**: Root mean square of the amplitudes in the interval (98% $X$, 102% $X$).

# Experimental Framework
In order to facilate research on this dataset, we provide an easy quick usage and customization of experiments in this dataset.
See notebook tutorial.ipynb for details.

## Requirements
The code was only tested in Python 3.8 and Ubuntu 20.04, but should work without any modifications in Python 3.7+ and/or Windows.
Pytorch 1.7+ with cuda is required for running the benchmark classifier Triplet network.
To install requirements, run
```bash
pip install -r requirements.txt
```


## How to run (.yaml configuration file)
The framework can be run using the script [test.py](test.py). The script takes a yaml configuration file as an argument, informing how the experiment should be done. Examples of this configuration file are found in the [configs](configs) directory.
These are the valid keywords insided the yaml file:
- cross_validation (dict): Determines the cross validation method (sampler) to be used. Valid keys:
    - class (str): The class name of the cross validation method. Valid options are: RepeatedStratifiedKFold, Predefinedkfold, StratifiedShuffleSplit, StratifiedKFold.
    - \*: Specific parameters of the sampler class should go here, at the same level as *class* keywork.
- feature_extractors (dict): Determines which feature extractors should be used. This is only necessary when doing metric learning or something similar. Keys inside this dict should be a path (separated by a '.') to a function or class responsible of creating a object that has 2 methods:
    ```python
    def fit(self, X, y):
        pass
        
    def transform(self, X):
        pass
    ```
    parameters of the function can be passed by specifying it as keys/values inside the definition of the feature extractor. Example:
    ```yaml
    - feature_extractors.create_MetricLearningNetwork:
        name: tripletnet
        loss_function: tripletloss
        loss_function__margin: 0.2
        loss_function__triplets_per_anchor: 20
        module__num_outputs: 8
        learning_rate: 1.0e-4
        max_epochs: 200
        batch_size: 200
    ```
    This configuration tells the framework to execute function `create_MetricLearningNetwork(module_num_outputs=8, learning_rate=1.0e-4, max_epochs=200, batch_size=200, loss_function='tripletloss', loss_function__margin=0.2, loss_function__triplets_per_anchor=20)` of module/file "feature_extractors" in order to create an object with `fit` and `transform` methods. The name is a special keyword that gives a prettier arbitrary name of your choice, instead of the default "feature_extractors.create_MetricLearningNetwork".
- base_classifiers (dict): Defines classifiers to be trained and tested. The keys and values here can be any sklearn classifier or any class that implements fit and predict methods. Example:
```yaml
  - sklearn.neighbors.KNeighborsClassifier:
      name: knn
      n_neighbors: 3
```

## Paper Experiments
To run the experiments of the paper, run the following command:
```bash
python test.py -i data --config configs/paper_experiments.yaml -o results.csv
```
