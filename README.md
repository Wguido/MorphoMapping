# MorphoMapping

The MorphoMapping repository contains the code and raw data used for our ‚xy‘ paper.  
The framework can be found in the 'MorphoMapping' folder.  
IFC data was acquired through the IDEAS® software.

# Description

![Untitled (2)](https://github.com/Wguido/MorphoMapping/assets/117764795/c555157a-0e44-4a75-8bff-ad45c663f2cc)

# Installation

It is recommended to choose conda as your package manager. Conda can be obtained, e.g., by installing the Miniconda distribution. For detailed instructions, please refer to the respective documentation.

With conda installed, open your terminal and create a new environment by executing the following commands::

    conda create -n morphomapping python=3.10
    conda activate morphomapping

## PyPI

Currently, morphomapping is in beta-phase. There will be a PyPI release once the beta phase is finished.

    pip install morphomapping


## Development Version

In order to get the latest version, install from [GitHub](https://github.com/Wguido/MorphoMapping) using
    
    pip install git+https://github.com/Wguido/MorphoMapping@main

Alternatively, clone the repository to your local hard drive via

    git clone https://github.com/Wguido/MorphMapping.git && cd MorphoMapping
    git checkout --track origin/main
    pip install .

Note that while MorphoMapping is in beta phase, you need to have access to the private repo.

## Jupyter notebooks

Jupyter notebooks are highly recommended due to their extensive visualization capabilities. Install jupyter via

    conda install jupyter

and run the notebook by entering `jupyter-notebook` into the terminal.


### Dependencies
* The following Python packages (Python V.3.9.15) are needed for MorphoMapping:
  
| Package | Version |
| --- | --- |
| `bokeh` | *2.4.3*  |
| `flowkit` | *0.9.3*  |
| `hdbscan` | *0.8.28*  |
| `matplotlib` | *3.6.2*  |
| `numpy` | *1.22.3*  |
| `pandas` | *1.5.2*  |
| `pandas-bokeh` | *0.5.5*  |
| `seaborn` | *0.12.1*  |
| `scikit-learn` | *1.0.2*  |
| `umap-learn` | *0.5.5*  |

* The following R libraries (R V.4.2.1)  are required for MorphoMapping:
  
| Library | Version |
| --- | --- |
| `ggplot2` | *3.4.4*  |
| `ggpubr` | *0.6.0*  |
| `here` | *1.0.1*  |
| `IFC` | *0.2.1*  |
| `rstatix` | *0.7.2*  |
