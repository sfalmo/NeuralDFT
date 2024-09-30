[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8380003.svg)](https://doi.org/10.5281/zenodo.8380003)


# Neural functional theory for inhomogeneous fluids: Fundamentals and applications

This repository contains code, data sets and models corresponding to the following publication:

**Neural functional theory for inhomogeneous fluids: Fundamentals and applications**  
*Florian Sammüller, Sophie Hermann, Daniel de las Heras, and Matthias Schmidt, [Proc. Natl. Acad. Sci. **120**, e2312484120](https://doi.org/10.1073/pnas.2312484120) (2023); [arXiv:2307.04539](https://arxiv.org/abs/2307.04539).*


### Setup

Working in a virtual environment is recommended.
Set one up with `python -m venv .venv`, activate it with `source .venv/bin/activate` and install the required packages with `pip install -r requirements.txt`.
To use a GPU with Tensorflow/Keras, refer to the corresponding section in the installation guide at [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).


### Instructions

Simulation data can be found in `data` and trained models are located in `models`.
A sample script for training a model from scratch is given in `learn.py`.
The usage of a trained model, e.g. for the self-consistent calculation of density profiles, is illustrated in `neuraldft.py`.
Some useful utilities are provided in `utils.py`, such as tools for functional calculus as well as data generators and callbacks for training.


### Further information

The reference data has been generated with grand canonical Monte Carlo simulations using [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd).
The analytic DFT calculations with fundamental measure theory have been performed with the Julia library [ClassicalDFT.jl](https://gitlab.uni-bayreuth.de/bt306964/ClassicalDFT.jl).
