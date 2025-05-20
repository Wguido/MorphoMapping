# MorphoMapping

[![Documentation][badge-docs]][link-docs]
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[badge-docs]: https://img.shields.io/readthedocs/morphomapping
 
MorphoMapping is an analytical framework designed for the analysis of Imaging Flow Cytometry (IFC) data. It is based on our Python package, morphomapping, which provides tools for dimensionality reduction and clustering of IFC data. Step 2 and Step 4 can be considered optional.

# Description

| workflow | details | 
| --- | --- |
| Step 1 | IFC Data Acquisation|
| Step 2 | Convert .daf to .FCS (R)|
| Step 3 | morphomapping (Python)|
| Step 4 | Cluster Analysis (R)|

# Getting started

Please refer to the [documentation][link-docs].

# Installation

It is recommended to choose conda as your package manager. Conda can be obtained, e.g., by installing the Miniconda distribution. For detailed instructions, please refer to the respective documentation.

With conda installed, open your terminal and create a new environment by executing the following commands::

    conda create -n morphomapping python=3.10
    conda activate morphomapping

## PyPI

Install morphomapping via the pypi release:

    pip install morphomapping


## Development Version

In order to get the latest version, install from [GitHub](https://github.com/Wguido/MorphoMapping) using
    
    pip install git+https://github.com/Wguido/MorphoMapping@main

Alternatively, clone the repository to your local hard drive via

    git clone https://github.com/Wguido/MorphoMapping.git && cd MorphoMapping
    git checkout --track origin/main
    pip install .

Note that while MorphoMapping is in beta phase, you need to have access to the private repo.

## Jupyter notebooks

Jupyter notebooks are highly recommended due to their extensive visualization capabilities. Install jupyter via

    conda install jupyter

and run the notebook by entering `jupyter-notebook` into the terminal.


## Dependencies
* The following Python packages are needed for MorphoMapping:
  
| Package | 
| --- | 
| `bokeh` | 
| `flowkit` | 
| `hdbscan` | 
| `matplotlib` | 
| `numpy` | 
| `openpyxl` | 
| `pandas` | 
| `scikit-learn` | 
| `umap-learn` | 
| `openpyxl` |
| `adjustText` |

* The following R libraries (R V.4.2.1)  are required for MorphoMapping's optional DAFtoFCS-Converter and the Cluster Analysis:
  
| Library | Version |
| --- | --- |
| `ggplot2` | *3.4.4*  |
| `ggpubr` | *0.6.0*  |
| `here` | *1.0.1*  |
| `IFC` | *0.2.1*  |
| `rstatix` | *0.7.2*  |


[link-docs]: https://morphomapping.readthedocs.io
