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
- MATLAB (2019b or newer)

## Repository Contents

- `matlab_demodulation_script.m`: Demodulates raw .lvm files and calculates dF/F
- `FiberPhotometryAnalysisTool/`: GUI-based tool for group analysis
- `fiberphotometry_example_dataset/`: Sample data for testing the pipeline

## Usage

### 1. Signal Demodulation

1. Edit `matlab_demodulation_script.m` to set `dirPath` to your .lvm files location
2. Run the script in MATLAB to process files

The script creates a folder structure containing:
- Demodulated signal files (_C.csv, _S.csv)
- dF/F calculations (_dff.csv)
- Visualization plots

### 2. Data Analysis

1. Install the analysis tool environment:
```bash
conda env create --name fp_analysis --file=FiberPhotometryAnalysisTool/requirements.yml
conda activate fp_analysis

2. Launch the GUI: 

cd FiberPhotometryAnalysisTool
python main.py
