# Fiber Photometry Analysis Scripts

## Description
This repository contains custom analysis scripts for fiber photometry data used in our paper "Linking Brain Circuitry and Neural Plasticity in Antidepressant Response: The mPFC-Reuniens-Hippocampus Pathway." 

## Contents
- `matlab_demodulation_script.m`: Matlab script for demodulating raw photometry signals
- `lvm_import2.m`: Function for importing data from a LabView LVM file
- `plotting.py`: Visualization tools for creating publication-quality figures
- `example_workflow.ipynb`: Jupyter notebook demonstrating the complete analysis pipeline

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter (for notebooks)

## Installation
Clone this repository:
```git clone https://github.com/yourusername/fiber-photometry-analysis```

Install required packages:
```pip install -r requirements.txt```

## Usage
Basic example of how to process fiber photometry data:

```python
import preprocessing as pp

# Load data
data = pp.load_data('path/to/your/data.csv')

# Apply motion correction and dF/F calculation
corrected_data = pp.motion_correct(data)
df_f = pp.calculate_df_f(corrected_data)
