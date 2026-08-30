{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Figure Reproduction\
\
This directory contains the Python scripts and supporting data files used to generate the figures presented in the associated MDPI paper.\
\
The files are provided to support the reproducibility and transparency of the reported results by allowing the figures in the manuscript to be regenerated from the corresponding experimental data.\
\
## Directory Structure\
\
    figure-reproduction/\
    \uc0\u9474 \
    \uc0\u9500 \u9472 \u9472  README.md\
    \uc0\u9474 \
    \uc0\u9500 \u9472 \u9472  heatmaps/\
    \uc0\u9474    \u9500 \u9472 \u9472  regen_heatmaps.py\
    \uc0\u9474    \u9500 \u9472 \u9472  9_nodes_unpoisoned.txt\
    \uc0\u9474    \u9492 \u9472 \u9472  9_nodes_4_poisoned_quantization.txt\
    \uc0\u9474 \
    \uc0\u9492 \u9472 \u9472  line-plots/\
        \uc0\u9500 \u9472 \u9472  regen_line_plots.py\
        \uc0\u9500 \u9472 \u9472  tabs.json\
        \uc0\u9492 \u9472 \u9472  pairs.json\
\
## Contents\
\
### `line-plots/`\
\
Contains the script and supporting data used to generate the main experimental line plots presented in the manuscript.\
\
The `regen_line_plots.py` script reads the experimental results and figure mappings from:\
\
- `tabs.json`\
- `pairs.json`\
\
It generates the figures associated with the different experimental configurations, including the analyses of noise, data imbalance, and poisoned nodes.\
\
The generated figures are saved under:\
\
    Figures_new/\
\
while preserving the output structure defined in `pairs.json`.\
\
### `heatmaps/`\
\
Contains the script and supporting data used to generate the threshold-sensitivity heatmaps.\
\
The `regen_heatmaps.py` script reads the node evaluation results from:\
\
- `9_nodes_unpoisoned.txt`\
- `9_nodes_4_poisoned_quantization.txt`\
\
It generates three heatmaps corresponding to:\
\
1. an unpoisoned configuration;\
2. four poisoned nodes without collusion; and\
3. four poisoned nodes with collusion.\
\
The generated files are:\
\
    threshold_sensitivity_unpoisoned.png\
    threshold_sensitivity_4_poisoned_non_colluding.png\
    threshold_sensitivity_4_poisoned_colluding.png\
\
Each heatmap reports the number of correctly classified models together with the false positive rate (FPR) and, when applicable, the false negative rate (FNR) for each threshold combination.\
\
The operating point used in the main experiments is also highlighted in the heatmaps.\
\
## Requirements\
\
The scripts require Python 3 and the following Python packages:\
\
- `matplotlib`\
- `numpy`\
\
The remaining dependencies (`json`, `os`, `re`, and `math`) are part of the Python standard library.\
\
Install the required packages with:\
\
    pip install matplotlib numpy\
\
## Usage\
\
### Line Plots\
\
Navigate to the `line-plots/` directory:\
\
    cd line-plots\
\
Then run:\
\
    python3 regen_line_plots.py\
\
The script will read `tabs.json` and `pairs.json` from the same directory and generate the corresponding figures.\
\
### Threshold-Sensitivity Heatmaps\
\
Navigate to the `heatmaps/` directory:\
\
    cd heatmaps\
\
Then run:\
\
    python3 regen_heatmaps.py\
\
The script will read the two experimental result files from the same directory and generate the three threshold-sensitivity heatmaps.\
\
## Figure Formatting\
\
The plotting configuration was designed for the figures included in the manuscript. The scripts use:\
\
- high-resolution output suitable for publication;\
- consistent dimensions across related figures;\
- font-size compensation for the dimensions used in the manuscript;\
- colorblind-friendly visualization;\
- distinct line styles and markers where applicable; and\
- additional non-color visual cues where necessary to preserve readability in grayscale.\
\
For the line plots, constant baselines are represented by horizontal lines without point markers, while the remaining experimental series use distinct markers and line styles.\
\
For the sensitivity heatmaps, successful configurations are identified both by their background shading and by an additional cell outline, so their interpretation does not depend exclusively on color.\
\
## Threshold-Sensitivity Analysis\
\
The sensitivity analysis evaluates combinations of weighted F1-score and consensus thresholds.\
\
The weighted F1-score thresholds evaluated are:\
\
    40%, 50%, 60%, 70%, 80%\
\
The consensus requirements evaluated for the nine-node configuration are:\
\
    4/9, 5/9, 6/9\
\
The heatmaps report:\
\
- the number of correctly classified models out of nine;\
- the false positive rate (FPR); and\
- the false negative rate (FNR), when poisoned models are present.\
\
The `70%` weighted F1-score threshold and `5/9` consensus requirement, corresponding to the operating point used in the main experiments, are highlighted with a dashed outline.\
\
## Reproducibility Scope\
\
These files reproduce the **figures from the experimental results supplied in this directory**.\
\
They do not perform the underlying model training or reproduce the complete experimental pipeline. Instead, they process the recorded experimental results and regenerate the visualizations reported in the manuscript.\
\
The purpose of this directory is therefore to make the generation of the published figures and threshold-sensitivity visualizations transparent and reproducible.\
\
## Associated Publication\
\
These scripts and supporting files accompany the experimental results reported in the associated MDPI publication.\
\
If you use this material, please cite the corresponding paper.\
\
## License\
\
Please refer to the repository's license file for the terms under which this code and the accompanying files may be used.}