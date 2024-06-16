# Protein-Molecule Interaction Classification

This project focuses on classifying interacting/non-interacting pairs of proteins and molecules using simple neural network. The model, provided here for making predictions, uses ProtT5 protein embeddings and Morgan fingerprints of chemicals as input. 
You can also obtain a model trained on other representations, like AlphaFold2 for proteins and MolecularTransformer for ligands (see notebook for training below). The project is implemented using PyTorch and PyTorch Lightning, and training experiments are tracked with Weights & Biases (W&B).

## Table of Contents
- [Getting Started](#getting-started)
- [Notebooks](#notebooks)
- [Parameters](#parameters)
- [Contributing](#contributing)

## Limitations
The model is only available for proteins with single peptide chains. It should be also noted that the length of proteins used for training did not exceed 1000 residues. 

## Getting Started

1. Open the provided Google Colab notebooks.

2. Make a copy of the notebook to your own Google Drive:
    - Click on `File` -> `Save a copy in Drive`.

3. Follow the instructions within the notebooks to run the cells and train the model or predict possibilities of protein-ligand interactions.

## Notebooks

The project contains the following notebooks:

- **[Reproduce the train and test experiment with this notebook](https://colab.research.google.com/drive/1iXF5kaBAN-kw2K2TBXxfaJqLqhdfexWK?usp=sharing)**: This notebook is used to train the protein-molecule interaction classifier using ProtT5 embeddings and Morgan fingerprints or MolTr embeddings.
- **[Predict interaction possibility on your data](path_to_your_notebook)**: This notebook is used to train the interaction classifier using AF data.

## Parameters

### General Parameters

- `morgan_fp`: (boolean) Use Morgan fingerprints for molecular representation.
- `learning_rate`: (float) Learning rate for the optimizer.
- `epochs`: (int) Number of training epochs.
- `project`: (string) W&B project name.
- `weight_decay`: (float) Weight decay for the optimizer.
- `batch_size`: (int) Batch size for data loading.

### Model-Specific Parameters

- `input_size_protein`: (int) Size of the input feature vector for proteins.
- `input_size_molecule`: (int) Size of the input feature vector for molecules.
- `fc1_layer_size_factor`: (int) Factor to reduce the size of the first fully connected layer.
- `fc2_layer_size_factor`: (int) Factor to reduce the size of the second fully connected layer.
- `dropout_rate`: (float) Dropout rate applied after each layer.

### AF Model-Specific Parameters

- `single_size`: (int) Size of the single input feature vector.
- `pair_size`: (int) Size of the pair input feature vector.
- `msa_size`: (int) Size of the MSA input feature vector.
- `molecule_size`: (int) Size of the molecule input feature vector.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
