# Protein-Metabolite Interaction Classification

This project focuses on classifying interacting/non-interacting pairs of proteins and small molecules using simple neural network and highly informative embeddings from pre-trained models as input. Those include **[ProtT5](https://github.com/agemagician/ProtTrans?tab=readme-ov-file)** and **[AlphaFold2](https://github.com/google-deepmind/alphafold)** representations of proteins, **[MolecularTransformer](https://github.com/mpcrlab/MolecularTransformerEmbeddings)** and Morgan fingerprints representations of chemicals. You can make a prediction or obtain a model trained on any of these representations (see notebooks below). The project is implemented using PyTorch and PyTorch Lightning, and training experiments are tracked with Weights & Biases (W&B).


## Table of Contents
- [Limitations](#limitations)
- [Getting Started](#getting-started)
- [Parameters](#parameters)
- [Data and model availability](#data-availability)
- [Contributing](#contributing)

## Limitations
The model is only available for proteins with single peptide chains. It should be also noted that the length of proteins used for training was no more than 1000 residues. However, the prediction can be made for proteins of any length as embeddings with fixed shapes are used. 

## Getting Started

### **Reproduce the experiment**

1. Open the **[Reproduce the experiment](https://colab.research.google.com/drive/1iXF5kaBAN-kw2K2TBXxfaJqLqhdfexWK?usp=sharing)** notebook.

2. Make a copy of the notebook to your own Google Drive:
    - Click on `File` -> `Save a copy in Drive`.

3. Follow the instructions within the notebook to train the model.

### **Make a prediction**

The model, provided here for making predictions, uses ProtT5 protein embeddings and Morgan fingerprints of chemicals as input. 

1. Obtain protein embedding using [embed_ProtT5.ipynb](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing#scrollTo=QMoeBQnUCK_E) from **[ProtTrans Github repository](https://github.com/agemagician/ProtTrans?tab=readme-ov-file)**. Set `per_residue = False`, `per_protein = True` and `sec_struct = False`. The output should have size of 1024. Download the resulting h5 file to use later.

2. Open the **[Predict interactions](https://colab.research.google.com/drive/1qQVgUTXtOQ7zyH6bHB0X16tY1O6nPO94?usp=sharing)** notebook.

3. Make a copy of the notebook to your own Google Drive:
    - Click on `File` -> `Save a copy in Drive`.

4. Upload the h5-file with ProtT5 embedding to Google Colab in any convenient way.

4. Follow the instructions within the notebook to predict possibility of protein-ligand interaction.

## Model architecture

We used a simple MLP classifier. After each layer batch normalization was used and ReLU activation function was applied.

<img src="./Untitledneur.svg">

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
- `fc1_layer_size_factor`: (int) Factor to reduce the size after the first fully connected layer.
- `fc2_layer_size_factor`: (int) Factor to reduce the size after the second fully connected layer.
- `dropout_rate`: (float) Dropout rate applied after each layer.

### AF2 Model-Specific Parameters

- `single_size`: (int) Size of the single input feature vector.
- `pair_size`: (int) Size of the pair input feature vector.
- `msa_size`: (int) Size of the MSA input feature vector.
- `molecule_size`: (int) Size of the molecule input feature vector.

## Data and model availability
Data and model can be found **[here](https://drive.google.com/drive/folders/1u9DwSNje2gX-N0QgxxFFHanw705gH29N?usp=sharing)**

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
