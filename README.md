# Protein-Molecule Interaction Classification

This project focuses on classifying interacting/non-interacting pairs of proteins and molecules using neural networks. The model uses ProtT5 protein embeddings and various molecular representations, such as Morgan fingerprints and MolTr embeddings. The project is implemented using PyTorch and PyTorch Lightning, and experiments are tracked with Weights & Biases (W&B).

## Table of Contents
- [Getting Started](#getting-started)
- [Notebooks](#notebooks)
- [Parameters](#parameters)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

1. Open the provided Google Colab notebooks.

2. Make a copy of the notebook to your own Google Drive:
    - Click on `File` -> `Save a copy in Drive`.

3. Ensure you have the necessary datasets uploaded to your Google Drive or accessible via a URL.

4. Follow the instructions within the notebooks to run the cells and train the model.

## Notebooks

The project contains the following notebooks:

- **[Train_Classifier_Notebook](path_to_your_notebook)**: This notebook is used to train the protein-molecule interaction classifier using ProtT5 embeddings and Morgan fingerprints or MolTr embeddings.
- **[AF_Classifier_Notebook](path_to_your_notebook)**: This notebook is used to train the interaction classifier using AF data.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
