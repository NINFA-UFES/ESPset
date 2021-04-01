This repository provides the ESPset dataset, a real-world dataset for vibration-based fault diagnosis of electric submersible pumps used on offshore oil exploration.
In addition to that, this repository also provides an experimental framework for adequately comparing research works based on the ESPset dataset and defines some benchmark classifiers.

# ESPset
The ESPset dataset is a collection of vibration signals acquired from accelerometers strategically attached onto the components of Eletrical Submersible Centrifugal Pumps (ESP).
An ESP belong to a class of equipment used in the extraction and exploration of oil and gas subject to severe working conditions.
An ESP system consists of a coupled set of one or more electric motors, pumps and  protectors.
For more details about the theory behind this dataset, please refer to the paper ?.

The dataset is provided in two files: spectrum.csv and features.csv.

## spectrum.csv
This csv file is a matrix of 6032 lines and 12103 columns, whose values are float numbers seperated by a ';'. 
Each line of this file contains the spectrum of a single vibration signal collected from a sensor at a specific test condition of the ESP.
Each value is the amplitude in inches per second (velocity) at a specific frequency.
Each signal is normalized by the rotation frequency in which the ESP operates, in such a way that the amplitude with respect to the rotation frequency is always at the ?th point/column of all signals.

A simple way to load this matrix in python:
```python
import numpy as np
signals = np.loadtxt('data/spectrum.csv', delimiter=';')
```

## features.csv
This csv file of 6033 lines (one line for each signal + a header), contains some features and the labels for all signals.
- component: The type of the component in which the sensor of the vibration signal is attached. Can be "Motor", "Pump" or "Protector".
- comp_number: The component number, in order to distinguish ESP with two components of the same type (e.g two pumps).
- axis: The axis in which the sensor attached. Can be "X" or "Y".
- position: The relative height in which the sensor is attached. A value of 1.0 means at the top of the component, and a value of 0.5 means at the middle of the component.
- flow or lrnumber: ?
- esp_id: The id of the ESP.
- label: The classification.
Let F be defined as the rotation frequency in which the BCS is operated. Each feature is defined as:
- median(3,5): Median of the amplitudes in the interval (3Hz, 5Hz);
- median(F-1,F+1) Median of the amplitudes in the interval (F-1Hz, F+1Hz);
- a: Coefficient a of the exponential regression of type e^(aX+b) where X is an array of equally separated frequencies from 5Hz to 19Hz.
- b: Coefficient b of the exponential regression of type e^{(aX+b)} in the interval (5Hz, 19Hz);
- rotation1x: Frequency of the highest amplitude in the interval (F-3Hz, F-0.2Hz);
- peak1x: Amplitude in rotation1x;
- peak2x: Amplitude in 2 rotation1x;
- rms(F-1,F+1) Root mean square of the amplitudes in the interval (F-1, F+1).
\end{itemize}

# Experimental Framework
In order to facilate research on this dataset, we provide an easy quick usage and customization of experiments in this dataset.
See notebook tutorial.ipynb for details.

# Benchmark classifiers
We provide some baseline classifiers and their respectives results as referential.
The configuration file for running the baseline classifiers is configs/benchmark.xml?.
1. Triplet Network + Random forest: Achieved an average macro F-measure of ?
2. Hand-crafted features + Random forest: Achieved an average macro F-measure of ?
