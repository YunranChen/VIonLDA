# VIonLDA


This project is application of Variational Inference on Latent Dirichlet Allocation. We mainly reproduce paper by Blei et al. 2003. There are three main parts: 1. Functions packaged inside package VIonLDA, which includes all the versions of function mentioned in the report. Including data simulation and Variational EM algorithm on LDA and smoothing LDA. 2. Tests. All the simulations we mentioned in the report. 3. Examples. We give application on real data analysis. 4.Parallel. For the parallel version not always work. We exclude it from our package.

# Install the package

Please make sure the package version is 2.7. 

%%bash

pip install --index-url https://test.pypi.org/simple/ VIonLDA

To install the latest version:

First uninstall:

pip uninstall --index-url https://test.pypi.org/simple/ VIonLDA

Then install the latest version:

%%bash

pip install -I --index-url https://test.pypi.org/simple/ VIonLDA


# Functions inside package VIonLDA

This package includes:

## Main Functions 

1. simulation_data()

Data simulation according to LDA model, the output would be a list of one-hot-coding matrix.

2. M_step_Vectorization()

Vectorizaiton Variational EM algorithm on LDA. Inside would use function E_step_Vectorization().

3.M_step_Smoothing()

Vectorization version of Variational EM algorithm on smoothing LDA. Inside would use function E_step_Smoothing(). 

4.mmse()

Evaluation function 

## Other functions 

1.M_step_Plain()

Plain version Variational EM algorithm on LDA. Inside would use function E_step_Plain().

2.M_step_Structure()

Structure1 version Variational EM algorithm on LDA. Inside would use function E_step_Structure(). This algorithm is based on Vectorization version. But using vector as input for E_step.

3.M_step_Realdata()

To avoid overflow when applied in real dataset. We set the float128 for the data type. A slightly change on Vectorization version. Inside would use function E_step_Realdata()

# Tests

Inside is .ipynb showing how to use this functions and reproduce the result on simulation data in the report. 

# Examples

Inside is .ipynb showing how to use these functions on real datasets and reproduce the result in the report.

# Parallel

Inside is .ipynb showing the parallel version of Variational EM on LDA. Make sure you $pip install ray before run it. Sometimes ray cannot run in VM. Please restart the VM. Because Parallel version would not always work. Depends on the condition of your computer. We decide to exclude this part from the package.

# Notice

- All the defaults of the function are set as the report suggest. 

- To reproduce our result, do include random seed 123.

- We expand more on the discussion in our report to firgure out a better data structure. See lda_ideal_data_structure_worse_behavior.ipynb. For the poor behavior, we do not include this part in our package and report. But it is worth discussing.

# Ideal Data Structure But Poor Behavior

As we mentioned in the discussion, we design a version for ideal data structure. We implemented matrix multiplicaiton, vectorization and broadcast in this version. We can see the speed is super fast. However, the core idea that VI work is the Variational parameter $\phi$, $\gamma$ converge. (make the KL distance as small as possible to approximate the probability density) So if we break this rule, the algorithm cannot converge and even intractable. We can see with the same initialization, 10 times iteration, the time for this algorithm is only half of the vectorization version we adopt in the report. However, the mse is a factor of 0.1 to 0.01 in scale worse than the vectorization version. 

 
