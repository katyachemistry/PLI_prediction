import torch.nn as nn
import torch
import torchmetrics
import pytorch_lightning as L  
import torchmetrics


class InteractionClassifier_ProtT5_based(nn.Module):
    def __init__(self, input_size_protein, input_size_molecule, fc1_layer_size_factor, fc2_layer_size_factor, dropout_rate=0):
        super().__init__()
        output_size_protein_1 = int(input_size_protein / fc1_layer_size_factor)
        self.protein_fc1 = nn.Linear(input_size_protein, output_size_protein_1)
        output_size_protein_2 = int(output_size_protein_1 / fc2_layer_size_factor)
        self.protein_fc2 = nn.Linear(output_size_protein_1, output_size_protein_2)
        output_size_molecule_1 = int(input_size_molecule / fc1_layer_size_factor)
        self.molecule_fc1 = nn.Linear(input_size_molecule, output_size_molecule_1)
        output_size_molecule_2 = int(output_size_molecule_1 / fc2_layer_size_factor)
        self.molecule_fc2 = nn.Linear(output_size_molecule_1, output_size_molecule_2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(output_size_protein_2 + output_size_molecule_2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.norm_prot1 = nn.BatchNorm1d(output_size_protein_1)
        self.norm_prot2 = nn.BatchNorm1d(output_size_protein_2)
        self.norm_mol1 = nn.BatchNorm1d(output_size_molecule_1)
        self.norm_mol2 = nn.BatchNorm1d(output_size_molecule_2)
        self.norm_all = nn.BatchNorm1d(64)

    def forward(self, protein, molecule):
        molecule = molecule.view(molecule.size(0), -1).to(torch.float32)
        protein = self.relu(self.norm_prot1(self.protein_fc1(protein)))
        protein = self.dropout(protein)
        protein = self.relu(self.norm_prot2(self.protein_fc2(protein)))
        protein = self.dropout(protein)
        molecule = self.relu(self.norm_mol1(self.molecule_fc1(molecule)))
        molecule = self.dropout(molecule)
        molecule = self.relu(   self.norm_mol2(self.molecule_fc2(molecule)))
        molecule = self.dropout(molecule)
        combined = torch.cat((protein, molecule), dim=1)
        x = self.relu(self.norm_all(self.fc1(combined)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class InteractionClassifier_AF2_based(nn.Module):

    def __init__(self, single_size, pair_size, msa_size, input_size_molecule, fc1_layer_size_factor, fc2_layer_size_factor, dropout_rate=0):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        output_size_single_1 = int(single_size / fc1_layer_size_factor)
        self.single_fc1 = nn.Linear(single_size, output_size_single_1)
        self.norm_single_1 = nn.BatchNorm1d(output_size_single_1)

        output_size_pair_1 = int(pair_size / fc1_layer_size_factor)
        self.pair_fc1 = nn.Linear(pair_size, output_size_pair_1)
        self.norm_pair_1 = nn.BatchNorm1d(output_size_pair_1)

        output_size_msa_1 = int(msa_size / fc1_layer_size_factor)
        self.msa_fc1 = nn.Linear(msa_size, output_size_msa_1)
        self.norm_msa_1 = nn.BatchNorm1d(output_size_msa_1)

        output_size_single_2 = int(output_size_single_1 / fc2_layer_size_factor)
        self.single_fc2 = nn.Linear(output_size_single_1, output_size_single_2)
        self.norm_single_2 = nn.BatchNorm1d(output_size_single_2)

        output_size_pair_2 = int(output_size_pair_1 / fc2_layer_size_factor)
        self.pair_fc2 = nn.Linear(output_size_pair_1, output_size_pair_2)
        self.norm_pair_2 = nn.BatchNorm1d(output_size_pair_2)

        output_size_msa_2 = int(output_size_msa_1 / fc2_layer_size_factor)
        self.msa_fc2 = nn.Linear(output_size_msa_1, output_size_msa_2)
        self.norm_msa_2 = nn.BatchNorm1d(output_size_msa_2)

        output_size_molecule_1 = int(molecule_size / fc1_layer_size_factor)
        self.molecule_fc1 = nn.Linear(molecule_size, output_size_molecule_1)
        self.norm_mol1 = nn.BatchNorm1d(output_size_molecule_1)

        output_size_molecule_2 = int(output_size_molecule_1 / fc2_layer_size_factor)
        self.molecule_fc2 = nn.Linear(output_size_molecule_1, output_size_molecule_2)
        self.norm_mol2 = nn.BatchNorm1d(output_size_molecule_2)

        self.fc1 = nn.Linear(output_size_single_2 + output_size_pair_2 + output_size_msa_2 + output_size_molecule_2, 64)
        self.norm_all = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, single, pair, msa, molecule):
        molecule = molecule.view(molecule.size(0), -1).to(torch.float32)

        single = self.relu(self.norm_single_1(self.single_fc1(single)))
        single = self.dropout(single)
        single = self.relu(self.norm_single_2(self.single_fc2(single)))
        single = self.dropout(single)

        pair = self.relu(self.norm_pair_1(self.pair_fc1(pair)))
        pair = self.dropout(pair)
        pair = self.relu(self.norm_pair_2(self.pair_fc2(pair)))
        pair = self.dropout(pair)

        msa = self.relu(self.norm_msa_1(self.msa_fc1(msa)))
        msa = self.dropout(msa)
        msa = self.relu(self.norm_msa_2(self.msa_fc2(msa)))
        msa = self.dropout(msa)

        molecule = self.relu(self.norm_mol1(self.molecule_fc1(molecule)))
        molecule = self.dropout(molecule)
        molecule = self.relu(self.norm_mol2(self.molecule_fc2(molecule)))
        molecule = self.dropout(molecule)

        combined = torch.cat((single, pair, msa, molecule), dim=1)

        x = self.relu(self.norm_all(self.fc1(combined)))
        x = self.dropout(x)

        x = self.fc2(x)

        return x



class Lit(L.LightningModule):
    def __init__(self, model, optimizer_kwargs,
                 criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.25])),
                 optimizer_class=torch.optim.AdamW):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.train_auroc = torchmetrics.AUROC(task="binary")
        self.valid_auroc = torchmetrics.AUROC(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.valid_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.valid_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.valid_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.valid_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        return optimizer

    def training_step(self, batch):
        prots, mols, labels = batch
        out = self.model(prots, mols)
        loss = self.criterion(out, labels)
        self.log("loss_on_train", loss, prog_bar=True)
        self.train_auroc.update(out, labels)
        self.train_accuracy.update(out, labels)
        self.train_recall.update(out, labels)
        self.train_precision.update(out, labels)
        self.train_f1.update(out, labels)
        return loss

    def validation_step(self, batch):
        prots, mols, labels = batch
        out = self.model(prots, mols)
        loss = self.criterion(out, labels)
        self.log("loss_on_val", loss, prog_bar=True)
        self.valid_auroc.update(out, labels)
        self.valid_accuracy.update(out, labels)
        self.valid_recall.update(out, labels)
        self.valid_precision.update(out, labels)
        self.valid_f1.update(out, labels)

    def on_train_epoch_end(self):
        self.log("AUROC/train", self.train_auroc.compute(), prog_bar=True)
        self.log("Accuracy/train", self.train_accuracy.compute(), prog_bar=True)
        self.log("Recall/train", self.train_recall.compute(), prog_bar=True)
        self.log("Precision/train", self.train_precision.compute(), prog_bar=True)
        self.log("F1/train", self.train_f1.compute(), prog_bar=True)
        self.train_auroc.reset()
        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("AUROC/valid", self.valid_auroc.compute(), prog_bar=True)
        self.log("Accuracy/valid", self.valid_accuracy.compute(), prog_bar=True)
        self.log("Recall/valid", self.valid_recall.compute(), prog_bar=True)
        self.log("Precision/valid", self.valid_precision.compute(), prog_bar=True)
        self.log("F1/valid", self.valid_f1.compute(), prog_bar=True)
        self.valid_auroc.reset()
        self.valid_accuracy.reset()
        self.valid_recall.reset()
        self.valid_precision.reset()
        self.valid_f1.reset()

    def test_step(self, batch):
        prots, mols, labels = batch
        out = self.model(prots, mols)
        loss = self.criterion(out, labels)
        self.log("loss_on_test", loss, prog_bar=True)
        self.test_auroc.update(out, labels)
        self.test_accuracy.update(out, labels)
        self.test_recall.update(out, labels)
        self.test_precision.update(out, labels)
        self.test_f1.update(out, labels)

    def on_test_epoch_end(self):
        self.log("AUROC/test", self.test_auroc.compute(), prog_bar=True)
        self.log("Accuracy/test", self.test_accuracy.compute(), prog_bar=True)
        self.log("Recall/test", self.test_recall.compute(), prog_bar=True)
        self.log("Precision/test", self.test_precision.compute(), prog_bar=True)
        self.log("F1/test", self.test_f1.compute(), prog_bar=True)
        self.test_auroc.reset()
        self.test_accuracy.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_f1.reset()



class Lit_AF(L.LightningModule):
    def __init__(
        self,
        model,
        optimizer_kwargs,
        criterion=nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1.25])),
        optimizer_class=torch.optim.AdamW,

    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.train_auroc = torchmetrics.AUROC(task="binary")
        self.valid_auroc = torchmetrics.AUROC(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")

        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.valid_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")

        self.train_recall = torchmetrics.Recall(task="binary")
        self.valid_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")

        self.train_precision = torchmetrics.Precision(task="binary")
        self.valid_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")

        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.valid_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

        return optimizer


    def training_step(self, batch):
        single, pair, msa, mol, labels = batch
        out = self.model(single, pair, msa, mol)
        loss = self.criterion(out, labels)
        self.log("loss_on_train", loss, prog_bar=True)
        self.train_auroc.update(out, labels)
        self.train_accuracy.update(out, labels)
        self.train_recall.update(out, labels)
        self.train_precision.update(out, labels)
        self.train_f1.update(out, labels)

        return loss

    def validation_step(self, batch):
        single, pair, msa, mol, labels = batch
        out = self.model(single, pair, msa, mol)
        loss = self.criterion(out, labels)
        self.log("loss_on_val", loss, prog_bar=True)
        self.valid_auroc.update(out, labels)
        self.valid_accuracy.update(out, labels)
        self.valid_recall.update(out, labels)
        self.valid_precision.update(out, labels)
        self.valid_f1.update(out, labels)

    def on_train_epoch_end(self):
        self.log("AUROC/train", self.train_auroc.compute(), prog_bar = True)
        self.log("Accuracy/train", self.train_accuracy.compute(), prog_bar = True)
        self.log("Recall/train", self.train_recall.compute(), prog_bar = True)
        self.log("Precision/train", self.train_precision.compute(), prog_bar = True)
        self.log("F1/train", self.train_f1.compute(), prog_bar = True)

        self.train_auroc.reset()
        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("AUROC/valid", self.valid_auroc.compute(), prog_bar = True)
        self.log("Accuracy/valid", self.valid_accuracy.compute(), prog_bar = True)
        self.log("Recall/valid", self.valid_recall.compute(), prog_bar = True)
        self.log("Precision/valid", self.valid_precision.compute(), prog_bar = True)
        self.log("F1/valid", self.valid_f1.compute(), prog_bar = True)

        self.valid_auroc.reset()
        self.valid_accuracy.reset()
        self.valid_recall.reset()
        self.valid_precision.reset()
        self.valid_f1.reset()

    def test_step(self, batch):
        single, pair, msa, mol, labels = batch
        out = self.model(single, pair, msa, mol)
        loss = self.criterion(out, labels)
        self.log("loss_on_test", loss, prog_bar=True)
        self.test_auroc.update(out, labels)
        self.test_accuracy.update(out, labels)
        self.test_recall.update(out, labels)
        self.test_precision.update(out, labels)
        self.test_f1.update(out, labels)

    def on_test_epoch_end(self):
        self.log("AUROC/test", self.test_auroc.compute(), prog_bar = True)
        self.log("Accuracy/test", self.test_accuracy.compute(), prog_bar = True)
        self.log("Recall/test", self.test_recall.compute(), prog_bar = True)
        self.log("Precision/test", self.test_precision.compute(), prog_bar = True)
        self.log("F1/test", self.test_f1.compute(), prog_bar = True)

        self.test_auroc.reset()
        self.test_accuracy.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_f1.reset()