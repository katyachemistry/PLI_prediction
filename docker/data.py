import torch
from torch.utils.data import Dataset
import numpy as np


# ----Extracting data from dataset


def get_ProtT5_data(dataset, morgan_fp=True):
    """
    Extract molecular fingerprints, protein sequences, and labels from a dataset.

    This function retrieves molecular representations, protein sequences, and labels from the provided dataset.
    Depending on the value of the `morgan_fp` parameter, it extracts either Morgan fingerprints or MolTr values for
    the molecular representations.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset containing the molecular and protein data. It should have the following columns:
        - 'MorganFP' or 'MolTr': molecular representations
        - 'ProtT5': protein sequences
        - 'label': labels
    
    morgan_fp : bool, optional
        If True, the function extracts the 'MorganFP' column for molecular representations.
        If False, it extracts the 'MolTr' column. Default is True.

    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing three numpy arrays:
        - mol: The molecular representations (either Morgan fingerprints or MolTr values).
        - prots: The protein sequences.
        - labels: The labels associated with the molecular-protein pairs.
    """
    if morgan_fp:
        mol = dataset.MorganFP.values
    else:
        mol = dataset.MolTr.values

    prots = dataset.ProtT5.values
    labels = dataset.label.values
    return mol, prots, labels



def get_AF_data(dataset, morgan_fp=True):
    """
    Converts dataset values to tensors with optional padding and choice of molecular representations.

    Args:
        dataset: The input dataset containing values to be converted.
        morgan_fp_embeddings (bool, optional): If True, use Morgan fingerprint embeddings for molecular representations. If False, use MolTr embeddings. Default is True.

    Returns:
        tuple: Containing lists of tensors for single, pair, msa, mol, and label values.
    """

    # Choose the appropriate molecular representation
    if morgan_fp:
        mol = [torch.tensor(x, dtype=torch.float32) for x in dataset.MorganFP.values]
    else:
        mol = [torch.tensor(x, dtype=torch.float32) for x in dataset.MolTr.values]

    # Convert label values to tensors
    label = [torch.tensor(x, dtype=torch.float32) for x in dataset.label.values]

    make_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

    # Apply the padding function to single, pair, and msa values
    single = [make_tensor(x) for x in dataset.AF_single.values]
    pair = [make_tensor(x) for x in dataset.AF_pair.values]
    msa = [make_tensor(x) for x in dataset.AF_MSA.values]

    return single, pair, msa, mol, label


# ----Creating torch datasets


class ProteinMoleculeDataset(Dataset):
    """
    A PyTorch Dataset for ProtT5 representations data.

    This dataset class is designed to handle protein sequences, molecular representations, and their associated labels.
    
    Parameters:
    -----------
    proteins : array-like
        A list or array of protein sequences.
    molecules : array-like
        A list or array of molecular representations (e.g., fingerprints).
    labels : array-like
        A list or array of labels associated with the protein-molecule pairs.
    
    Attributes:
    -----------
    proteins : array-like
        The stored protein sequences.
    molecules : array-like
        The stored molecular representations.
    labels : torch.Tensor
        The stored labels as a tensor of shape (N, 1), where N is the number of samples.
    
    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.
    __getitem__(idx):
        Returns the protein sequence, molecular representation, and label for the sample at the given index.
    """
    def __init__(self, proteins, molecules, labels):
        self.proteins = proteins
        self.molecules = molecules
        self.labels = torch.tensor(np.vstack(labels), dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        protein = self.proteins[idx]
        molecule = self.molecules[idx]
        label = self.labels[idx]
        return protein, molecule, label


class ProteinMoleculeDataset_AF(Dataset):
    """
    A PyTorch Dataset for AF2 representations data.

    This dataset class handles AF2 data single, pairwise features, 
    multiple sequence alignments features, molecular representations, and their associated labels.
    
    Parameters:
    -----------
    single : array-like
        A list or array of single sequence features for the proteins.
    pair : array-like
        A list or array of pairwise features for the proteins.
    msa : array-like
        A list or array of multiple sequence alignments (MSAs) for the proteins.
    molecules : array-like
        A list or array of molecular representations (e.g., fingerprints).
    labels : array-like
        A list or array of labels associated with the protein-molecule pairs.
    
    Attributes:
    -----------
    single : array-like
        The stored single representations.
    pair : array-like
        The stored pairwise representations.
    msa : array-like
        The stored MSA representations.
    molecules : array-like
        The stored molecular representations.
    labels : torch.Tensor
        The stored labels as a tensor of shape (N, 1), where N is the number of samples.
    
    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.
    __getitem__(idx):
        Returns the single sequence features, pairwise features, multiple sequence alignments, molecular representation,
        and label for the sample at the given index.
    """
    def __init__(self, single, pair, msa, molecules, labels):
        self.single = single
        self.pair = pair
        self.msa = msa
        self.molecules = molecules
        self.labels = torch.tensor(np.vstack(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        single = self.single[idx]
        pair = self.pair[idx]
        msa = self.msa[idx]
        molecule = self.molecules[idx]
        label = self.labels[idx]
        return single, pair, msa, molecule, label
