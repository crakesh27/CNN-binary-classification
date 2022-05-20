# CNN
Simple keras implementation of CNN for binary classification.

## Requirements
* python
* numpy
* pandas
* sklearn
* keras
* matplotlib
* mpl_toolkits

## CNN
code file: `cnn.py`

data file: `data_for_cnn.mat`

label file: `class_label_.mat`

The dataset in ‘data_for_cnn.mat’ consists of 1000 ECG signals and each row corresponds to one ECG signal. The class label for each ECG signal is given in ‘class_label. mat’ file.

Convolution neural network for the binary classification.

Network flow:
Input-Convolution Layer-Pooling layer-FC1-FC2-Output

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.