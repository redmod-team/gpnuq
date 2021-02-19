[![DOI](https://zenodo.org/badge/168945305.svg)](https://zenodo.org/badge/latestdoi/168945305)
[![Documentation Status](https://readthedocs.org/projects/profit/badge/?version=latest)](https://profit.readthedocs.io/en/latest/?badge=latest)

<img src="logo.png" width="208.5px">

## Probabilistic Response Model Fitting with Interactive Tools

This is a collection of tools for studying parametric dependencies of 
black-box simulation codes or experiments and construction of reduced 
order response models over input parameter space. 

proFit can be fed with a number of data points consisting of different 
input parameter combinations and the resulting output of the model under 
investigation. It then fits a response "surface" through the point cloud.
This probablistic response model allows to predict ("interpolate") the output 
at yet unexplored parameter combinations including uncertainty estimates. 
It can also tell you where to put more training points to gain maximum new 
information (experimental design) and automatically generate and start
new simulation runs locally or on a cluster. Results can be explored and checked 
visually in a web frontend.

Telling proFit how to interact with your existing simulations is easy
and requires no changes in your existing code. Current functionality covers 
uncertainty quantification via polynomial chaos expansion 
with [chaospy](https://github.com/jonathf/chaospy) as a backend. Support for 
response surface / surrogate models via 
[GPflow](https://github.com/GPflow/GPflow) is under development. 
The web frontend is based on [plotly/dash](https://github.com/plotly/dash).

## Features

* Compute evaluation points to run simulation for UQ (full or sparse grid)
* Template replacement and automatic generation of run directories
* Starting parallel runs locally or on the cluster (SLURM)
* Collection of result output and postprocessing with UQ

## Installation

Currently the code is under heavy development so it should be cloned 
from GitHub via Git and pulled regularily. 

### Dependencies
* PyYAML
* ChaosPy
* GPFlow 2.0
* pyccel

All dependencies should be installed automatically when using `pip`.

### Installation from Git
To install proFit for the current user (`--user`) in development-mode (`-e`) use:

```bash
git clone https://github.com/redmod-team/profit.git
cd profit
pip install -e . --user
```

### Options
* Disable compilation of the fortran modules

        pip install . --global-option "--no-fortran"

* Install requirements for building the documentation using `sphinx`

        pip install .[docs]        

## HowTo

Examples for different model codes are available under `examples/`:
* `fit`: Simple fit via python interface.
* `mockup`: Simple model called by console command based on template directory.


1. Create and enter a directory `study` containing `profit.yaml` for your run.
   If your code is based on text configuration files for each run, copy the according directory to `template` and replace values of parameters to be varied within UQ/surrogate models by placeholders `{param}`.
   
2. Preprocessing:  
   ```
   profit pre
   ```
   to generate points where model is evaluated, and possibly run directories based on `template`.
   Evaluation points are stored inside `input.txt`
  
3. Running model: 
   ```
   profit run
   ```
   to start simulations at all the points. If `run.backend` is of type `PythonFunction`, results
   of model runs are stored in `output.txt` already here. Otherwise output is stored in model code-specific format.
  
4. Collect results: 
   ```
   profit collect
   ```
   to collect output from the runs into `output.txt`.
   

5. Explore data graphically: 
   ```
   profit ui
   ```
   starts a Dash-based browser UI

Fitting and UQ routines are currently being refactored and available via the Python API.
  
## User-supplied files

* `profit.yaml`
  * Add parameters and their distributions via `input`
  * Specify names of outputs in `output`
  * Set `run.backend` to a class available inside `profit.run`
  
* `interface.py`
  * `get_output()` should return model output as a numpy array in the order and shape specified in `profit.yaml`.
    The current path is the respective run directory. Can be skipped if `run.backend` is of type `PythonFunction`.
