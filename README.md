# Augmented-Neural-Network

Augmented Neural Network (NN+C) is a lightweight performance prediction model designed to accurately predict the execution time of a kernel operation on an arbitrary platform with arbitrary implementations, using a small amount of training time.

The key idea of NN+C is to utilize known mathematical function *f(K, H)* as an extra input to NN. For example, in matrix-matrix multiplication, besides using basic features such as matrix dimensions, matrix density as inputs, we calculate the number of total operations during matrix-matrix multiplication, that is, *f(K, H) = m\*n\*k*. 

The lightweight aspect enables fast decision making during compile-time as well as run-time. NN+C provides the flexibility to incorporate any tunable parameter available for the kernel and the hardware. 

For any kernel-variant-hardware combination, one NN+C is needed for one kernel regardless of any variant-hardware combination, providing high portability. 

---

This repository contains:
* `data_generation`: files used to generate training and testing dataset
* `performance_prediction`: performance prediction models namely NN+C and baselines

## Prerequisites

for **data generation**:
* [Eigen](http://eigen.tuxfamily.org/) >= 3.3.7
* [Boost](https://www.boost.org/) >= 1.71.0

for **performance prediction**:
* python >= 3.6.5
* tensorflow >= 1.8.0
* numpy >= 1.16.1
* pandas >= 0.23.0
* scikit-learn >= 0.22.2
* scipy >= 0.16.0


## Building

Code (`.cpp` and `.cu` files) in `data_generation` has to be built.

Note that:
* `<kernel> = {MM, MV, MP, MC}` 
* `<variant> = {eigen, boost, global, shared}`
* `<hardware> = {cpu, gpu}`

### On CPU

#### Eigen

`<kernel>_C_eigen.cpp` can be compiled by:

```
% clang++ -Xpreprocessor -fopenmp -std=c++11 -lomp <kernel>_C_eigen.cpp -o *
```

Or 

```
% g++ -g -Wall -fopenmp -std=c++11 <kernel>_C_eigen.cpp -o *
```

#### Boost

`<kernel>_C_boost.cpp` can be compiled by:

```
% g++ -g -Wall -std=c++11 -DNDEBUG -DBOOST_UBLAS_NDEBUG <kernel>_C_boost.cpp -o *
```

### On GPU

`<kernel>_G_<variant>.cu` can be compiled by:

```
% nvcc -std=c++11 <kernel>_G_<variant>.cu -o *
```

## Usage

### generate data

```
% cd data_generation/<hardware>/<variant>
% ./<executable>
```

**output filename**, **number of data instances to generate**, (**number of threads to utilize**) can be customized in `<kernel>_C_<variant>.cpp` or `<kernel>_G_<variant>.cu`


### predict performance

```
% mv data_generation/<hardware>/<variant>/<output_filename> performance_prediction/<hardware>/<variant>/<output filename>
% python *
```

## Acknowledgments

* This work is supported by the Defense Advanced Research Projects Agency (DARPA), under the Software Defined Hardware (SDH) project, at the University of Southern California.
* This work is supported by the Defense Advanced Research Projects Agency (DARPA), under the Performant Automation of Parallel Program Assembly (PAPPA) project, at the University of Southern California.
