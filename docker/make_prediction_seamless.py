import argparse
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from rdkit import Chem
from transformers import T5EncoderModel, T5Tokenizer

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_T5_model():
    """Load ProtT5 model and tokenizer."""
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # Move model to GPU
    model = model.eval()  # Set model to evaluation mode
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return model, tokenizer

def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=1000, max_batch=100):
    """Generate protein embeddings."""
    results = {"protein_embs": dict()}
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_protein:
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    return results

def main():
    """Main function to execute the script."""
    from models import InteractionClassifier_ProtT5_based

    # Argument parsing
    parser = argparse.ArgumentParser(description="Protein and molecule interaction prediction")
    parser.add_argument('--smiles', type=str, required=True, help="SMILES string of the molecule")
    parser.add_argument('--sequence', type=str, required=True, help="Amino acid sequence of the protein")
    parser.add_argument('--protein_name', type=str, required=True, help="Name of the protein")
    args = parser.parse_args()

    # Load ProtT5 model and tokenizer
    prott5_model, tokenizer = get_T5_model()

    # Parameters from command-line arguments
    SMILES = args.smiles
    sequence = args.sequence
    protein_name = args.protein_name
    protein_seq_dict = {protein_name: sequence}

    # Load and prepare model
    checkpoint_path = 'ProtT5_Morgan7.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = InteractionClassifier_ProtT5_based(1024, 1024, 2, 2, 0)
    new_state_dict = {}
    for key in list(checkpoint['state_dict'].keys())[:-1]:
        new_state_dict[key[6:]] = checkpoint['state_dict'][key]
    model.load_state_dict(new_state_dict)
    model.eval()

    # Generate embeddings
    results = get_embeddings(prott5_model, tokenizer, protein_seq_dict, per_residue=False, per_protein=True, sec_struct=False)

    # Generate Morgan Fingerprint and make prediction
    protein = torch.tensor(results["protein_embs"][protein_name][:]).to(torch.float32).view(1, 1024)
    mol = Chem.MolFromSmiles(SMILES)
    fpts = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    mfpts = torch.tensor(fpts).to(torch.float32).view(1, 1024)

    with torch.no_grad():
        proba = float(nn.functional.sigmoid(model(protein, mfpts))[0][0])
    print(f'The probability of this interaction is {proba:.3f}')

if __name__ == "__main__":
    main()
