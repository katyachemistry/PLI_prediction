import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py
from rdkit.Chem import AllChem
from rdkit import Chem

from models import InteractionClassifier_ProtT5_based

def main():
    parser = argparse.ArgumentParser(description='Protein-ligand interaction prediction')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string of the molecule')
    parser.add_argument('--protein_name', type=str, required=True, help='Protein name as it was written in the FASTA-file')
    parser.add_argument('--path_to_protT5_h5', type=str, required=True, help='Path to the obtained ProtT5 embeddings')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda. Default is cpu')

    
    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    fpts = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    mfpts = torch.tensor(fpts).to(torch.float32).view(1, 1024)

    checkpoint_path = 'ProtT5_Morgan7.ckpt'
    if args.device=='cpu':
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    model = InteractionClassifier_ProtT5_based(1024, 1024, 2, 2, 0)
    new_state_dict = {}
    for key in list(checkpoint['state_dict'].keys())[:-1]:
        new_state_dict[key[6:]] = checkpoint['state_dict'][key]
    model.load_state_dict(new_state_dict)
    model.eval()

    protein = torch.tensor(h5py.File(args.path_to_protT5_h5, 'r')[args.protein_name][:]).to(torch.float32).view(1, 1024)

    with torch.no_grad():
        proba = float(nn.functional.sigmoid(model(protein, mfpts))[0][0])
    print(f'The probability of this interaction is {proba:.3f}')

if __name__ == "__main__":
    main()
