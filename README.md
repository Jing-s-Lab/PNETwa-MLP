# PNETwa-MLP
# Table of Contents
- [Introduction](#introduction)
- [Install model](#install-model)
  - [Step 1](#step-1)
  - [Step 2](#step-2)
- [ Training model](#training-model)
  - [Step 1](#step-1-1)
  - [Step 2](#step-2-1)
  - [Step 3](#step-3)
  - [Step 4](#step-4)
  - [Step 5](#step-5)
  - [Step 6](#step-6)

# Introduction
In this study, we propose a novel approach that combines the incorporation of biological information with a Graph Neural Network called PNETwa and an MLP. PNETwa represents a type of Graph Neural Network that possesses the ability to learn node-level and graph-level representations while considering the topological relationships between nodes. The MLP, on the other hand, is a commonly used feedforward neural network model known for its powerful nonlinear modeling capabilities. Together, they can learn more comprehensive representations of global features from the node-level or graph-level representations extracted by the GNN.

By incorporating the relationships between genes and pathways from bioinformatics into PNETwa, we are able to extract feature genes related to prostate cancer metastasis from whole-genome sequencing data. These feature genes are then inputted into the MLP to achieve the prediction of prostate cancer metastasis. Our method achieves an AUC of 0.90 and an F1 score of 0.84 in data from Western populations, and an AUC of 0.89 and an F1 score of 0.74 in mixed data from Western and Asian populations.

Overall, this research contributes to the effective resolution of complex issues in primary and metastatic detection in clinical settings, opening up new possibilities in this field.

# Install model
## Step 1
Create a PNETwa environment：(ubuntu)

```bash
cd ./PNETwa
conda env create --name PNETwa --file=environment.yml
```

## Step 2
Create a MLP environment：(windows)

```bash
cd ./MLP
conda env create --name MLP --file=environment.yml
```

# Training model
## Step 1
Activate the PNETwa environment:

```bash
cd ./PNETwa
source activate PNETwa
```
## Step 2
Train PNETwa:
```bash
export PYTHONPATH=~/pnet_prostate_paper:$PYTHONPATH
cd ./PNETwa/train
python run_me.py
```
## Step 3
Get node contribution:
```bash
cd ./PNETwa/analysis
python run_it_all.py
```
This will generate 'node_importance_graph_adjusted.csv' in './PNETwa/analysis/extracted/'
## Step 4
To extract the contribution magnitude of genes (coef_combined column) from 'node_importance_graph_adjusted.csv', follow these steps:

1. Open the file 'node_importance_graph_adjusted.csv'
2. Locate the column named "coef_combined," which represents the contribution magnitude of genes.
3. Retrieve the values from the coef_combined column, which indicate the respective contribution magnitudes of genes.

Here is an example Python code snippet that demonstrates how to extract the values from the "coef_combined" column using the pandas library:

```python
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("node_importance_graph_adjusted.csv")

# Extract the values from the "coef_combined" column
coef_combined_values = df["coef_combined"].values

# Print the values
print(coef_combined_values)
```
## Step 5
Activate the MLP environment:
```bash
conda activate MLP
```
## Step 6
Train MLP:
```bash
cd ./MLP
python model.py
```
To modify the feature genes in the 'ls' function in the model.py file, you can follow these steps:

1. Open the model.py file in a text editor or integrated development environment (IDE).
2. Locate the 'ls' function within the file.
3. Identify the feature genes that you want to modify. These may be variables, parameters, or other elements that affect the behavior of the function.
4. Save the modified file.
