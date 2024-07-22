import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse


# Load data
df = pd.read_pickle('./dataframe.pkl')

# Create dataset
np.random.seed(100)
prots_val_test = np.random.choice(df.STITCH_protein_ID.unique(), 1600, replace=False)

val_test_df = df.loc[df['STITCH_protein_ID'].isin(prots_val_test)]
df_dupl = pd.concat([df, val_test_df])
df_dupl['Duplicated'] = df_dupl.duplicated(['STITCH_protein_ID', 'SMILES'], keep=False)
train = df_dupl[~df_dupl['Duplicated']]
val = val_test_df.sample(frac=0.5)
test = val_test_df.drop(val.index)


def train_func(project, protein_representations, molecule_representations, epochs=1, batch_size=32, learning_rate=1e-5, weight_decay=0.01, dropout_rate=0,
          fc1_layer_size_factor=2, fc2_layer_size_factor=2):
    
    optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}

    if molecule_representations=='morgan':
        morgan_fp=True
        input_size_molecule = 1024
    else:
        morgan_fp=False
        input_size_molecule = 512
    

    if protein_representations=='prott5':

        from models import InteractionClassifier_ProtT5_based, Lit

        from data import get_ProtT5_data, ProteinMoleculeDataset

        test_mols, test_prots, test_labels = get_ProtT5_data(test, morgan_fp=morgan_fp)
        val_mols, val_prots, val_labels = get_ProtT5_data(val, morgan_fp=morgan_fp)
        train_mols, train_prots, train_labels = get_ProtT5_data(train, morgan_fp=morgan_fp)

        input_size_protein = 1024

        model = InteractionClassifier_ProtT5_based(input_size_protein, input_size_molecule, fc1_layer_size_factor, fc2_layer_size_factor, dropout_rate)

        lit_model = Lit(model, optimizer_kwargs)

        train_dataset = ProteinMoleculeDataset(train_prots, train_mols, train_labels)
        val_dataset = ProteinMoleculeDataset(val_prots, val_mols, val_labels)
        test_dataset = ProteinMoleculeDataset(test_prots, test_mols, test_labels)


    elif protein_representations=='af2':

        from models import InteractionClassifier_AF2_based, Lit_AF

        from data import get_AF_data, ProteinMoleculeDataset_AF

        test_single, test_pair, test_msa, test_mol, test_label = get_AF_data(test, morgan_fp=morgan_fp)
        val_single, val_pair, val_msa, val_mol, val_label = get_AF_data(val, morgan_fp=morgan_fp)
        train_single, train_pair, train_msa, train_mol, train_label = get_AF_data(train, morgan_fp=morgan_fp)

        single_size = 256
        pair_size = 128
        msa_size = 23

        model = InteractionClassifier_AF2_based(single_size=single_size, pair_size=pair_size,
                                       msa_size=msa_size, input_size_molecule=input_size_molecule,
                                            fc1_layer_size_factor=fc1_layer_size_factor,
                                         fc2_layer_size_factor=fc2_layer_size_factor, dropout_rate=dropout_rate)
        
        lit_model = Lit_AF(model, optimizer_kwargs)

        train_dataset = ProteinMoleculeDataset_AF(train_single, train_pair, train_msa, train_mol, train_label)
        val_dataset = ProteinMoleculeDataset_AF(val_single, val_pair, val_msa, val_mol, val_label)
        test_dataset = ProteinMoleculeDataset_AF(test_single, test_pair, test_msa, test_mol, test_label)




    # Логин в WandB

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    wandb.login()
    wandb_logger = WandbLogger(project=project, log_model=True)


    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        log_every_n_steps=1,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        accelerator="cpu",
        devices=1
    )

    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader)
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--project", type=str, required=True, help="Name of the project")
    parser.add_argument("--protein-reps", type=str, required=True, help="prott5 for ProtT5 or af2 for AlphaFold2")
    parser.add_argument("--molecule-reps", type=str, default='morgan', help="morgan for Morgan Fingerprints or moltr for MolecularTransformer (default morgan)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--dropout-rate", type=float, default=0, help="Dropout rate (default: 0)")
    parser.add_argument("--fc1-layer-size-factor", type=int, default=2, help="Factor for FC1 layer size (default: 2)")
    parser.add_argument("--fc2-layer-size-factor", type=int, default=2, help="Factor for FC2 layer size (default: 2)")

    args = parser.parse_args()

    train_func(args.project, args.protein_reps, args.molecule_reps,
          args.epochs, args.batch_size, args.learning_rate, args.weight_decay,
          args.dropout_rate, args.fc1_layer_size_factor, args.fc2_layer_size_factor)

