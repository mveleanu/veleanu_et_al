# Fiber Photometry Analysis - Veleanu et al. (2025)

Code and examples for the fiber photometry analysis featured in Figure 3 of [Linking Brain Circuitry and Neural Plasticity in Antidepressant Response: The mPFC-Reuniens-Hippocampus Pathway](https://www.researchsquare.com/article/rs-6348176/v1)).

## Overview

This repository contains the pipeline used to analyze calcium imaging data from fiber photometry recordings in the ventral hippocampus. The analysis reveals how infralimbic cortex stimulation bidirectionally modulates hippocampal activity during behavioral transitions in the tail suspension test.

## Requirements

- MATLAB (2019b or newer)
- Python 3.8+ with dependencies listed in `requirements.yml`

## Repository Contents

- `matlab_demodulation_script.m`: Demodulates raw .lvm files and calculates dF/F
- `FiberPhotometryAnalysisTool/`: GUI-based tool for analyzing calcium dynamics 
- `example_data/`: Sample data for testing the pipeline

## Usage

### 1. Signal Demodulation

1. Edit `matlab_demodulation_script.m` to set `dirPath` to your .lvm files location
2. Run the script in MATLAB to process files

For example:
```matlab
% Change this path to your .lvm files location
dirPath = 'C:\Users\yourusername\Documents\fiber_photometry_data\';


The script processes files like D4_92_TST.lvm and creates a folder structure:
D4_92_TST/
├── guppy/
│   ├── D4_92_TST_C.csv   (control channel)
│   ├── D4_92_TST_S.csv   (signal channel)
│   └── output/
│       └── D4_92_TST_dff.csv  (calculated dF/F)
└── plot/
    ├── D4_92_TST.lvm_Periodogram.png
    ├── D4_92_TST.lvm_demodulated.png
    └── D4_92_TST.lvm_DFF.png

### 2. Data Analysis

1. Install the analysis tool environment:

conda env create --name fp_analysis --file=FiberPhotometryAnalysisTool/requirements.yml
conda activate fp_analysis

2. Launch the GUI: 

cd FiberPhotometryAnalysisTool
python main.py

3. In the GUI: 

Click "Add Group" to create experimental groups (e.g., "D4" for Day 4, "D6" for Day 6)
Add animals to each group
For each animal, specify:

Name: Animal ID (e.g., "092")
dF/F file path: (e.g., select the "D4_92_TST_dff.csv" file)
Column name: Select "data"
Timestamp file path: Select the corresponding timestamp file (e.g., select the 'D4_092_TST_Struggle.csv' file)
Column name: Select "Start"


Click "Analyze" to process the data
Click "Export Data" to save the results


## Example Data

The example_data/ folder contains:

- "raw" folder containing the source files
- "processed" folder containing the analysis steps
- "timestamps" folder, containing the timestamps 
- " results" folder, containing the output .csv files.


