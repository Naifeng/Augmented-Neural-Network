# Augmented-Neural-Network
Datasets, codes and binary files for Augmented Neural Network (NN+C)

## Getting Started

These instructions will assist you to run our experiments on your local machine for developing and testing purposes. 
* **Data-Generation**: codes (and binary files) used to generate training/testing dataset
* **Datasets**: datasets generated from data-generation
* **Models**: lightweight deep learning models implementing Augmented Neural Network


## Built With

* [TensorFlow](https://www.tensorflow.org/)(1.8.0) - Machine Learning Library

### Prerequisites

Download Eigen library for **Data-Generation** from:

* [Eigen](https://eigen.tuxfamily.org/dox/)

Install/update following python packages for **Models**:

```
pip install pandas
pip install numpy
pip install scipy
pip install -U scikit-learn
```

### Compiling

**Data-Generation** codes on CPU can be compiled by:

```
g++ -g -Wall -fopenmp -fopenmp -std=c++11
```

Or 

```
clang++ -Xpreprocessor -fopenmp -std=c++11 -lomp
```

**Data-Generation** codes on GPU can be compiled by:

```
nvcc
```

## Acknowledgments

This work is supported by Defense Advanced Research Projects Agency (DARPA), under the Software Defined Hardware (SDH) project, at University of Southern California.
