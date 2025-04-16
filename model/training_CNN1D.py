import os
import torch
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from argparse import ArgumentParser
from dataset import SpectrumDataset1D
from model import CNN1D
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

seed = 32
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SpectrumModel(pl.LightningModule):
    def __init__(self, in_channel, out_class, dropout=0.2):
        super(SpectrumModel, self).__init__()
        self.save_hyperparameters()
        self.model = CNN1D(in_channel, out_class, dropout)
        self.criterion = nn.CrossEntropyLoss() 
    
    def forward(self, x):
        logits = self.model(x)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        return probabilities

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def shared_evaluation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        return {"loss": loss, "probs": torch.nn.functional.softmax(y_hat, dim=1), "labels": y}

    def test_step(self, batch, batch_idx):
        return self.shared_evaluation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_evaluation_step(batch, batch_idx)

    def shared_evaluation_epoch_end(self, outputs, prefix):
        loss_key = "loss"
        probs_key = "probs"
        labels_key = "labels"

        avg_loss = torch.stack([x[loss_key] for x in outputs]).mean()
        preds_probs = torch.cat([x[probs_key] for x in outputs])
        labels = torch.cat([x[labels_key] for x in outputs])

        accuracy = accuracy_score(labels.cpu().numpy(), torch.argmax(preds_probs, dim=1).cpu().numpy())
        precision = precision_score(labels.cpu().numpy(), torch.argmax(preds_probs, dim=1).cpu().numpy(), average="macro")
        recall = recall_score(labels.cpu().numpy(), torch.argmax(preds_probs, dim=1).cpu().numpy(), average="macro")
        f1 = f1_score(labels.cpu().numpy(), torch.argmax(preds_probs, dim=1).cpu().numpy(), average="macro")

        self.log(f"{prefix}_loss", avg_loss)
        self.log(f"{prefix}_accuracy", accuracy)
        self.log(f"{prefix}_precision", precision)
        self.log(f"{prefix}_recall", recall)
        self.log(f"{prefix}_f1", f1)

        cm = confusion_matrix(labels.cpu().numpy(), torch.argmax(preds_probs, dim=1).cpu().numpy())
        print(f"{prefix} Confusion Matrix:")
        print(cm)

        labels_bin = label_binarize(labels.cpu().numpy(), classes=list(range(0, self.hparams.out_class)))
        preds_bin = torch.cat([x[probs_key] for x in outputs]).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(labels_bin.ravel(), preds_bin.ravel())
        roc_auc = auc(fpr, tpr)
        print(f"{prefix} ROC AUC:", roc_auc)


        output_file = "roc_curve_cnn1d.csv"
        roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
        roc_data.to_csv(output_file, index=False)

        metrics_file = f"{prefix}_metrics.csv"
        metrics_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)

        self.show_confusion_matrix(cm, f"{prefix}_confusion_matrix_1D.svg")
        self.show_roc_curve(fpr, tpr, roc_auc, f"{prefix}_roc_curve_1D.svg")


    def validation_epoch_end(self, outputs):
        self.shared_evaluation_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.shared_evaluation_epoch_end(outputs, "test")

    def show_confusion_matrix(self, cm, filename=None):
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        classes = ['Armid', 'CA', 'Cotton', 'Feather', 'Flax',
                'PA', 'PE', 'PET', 'PLA', 'PP', 'PU', 'PVA', 'SC',
                'Silk', 'Wool']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=18, fontproperties='Times New Roman', ha='right')
        plt.yticks(tick_marks, classes, fontproperties='Times New Roman', fontsize=18)

        plt.ylabel('True label', fontproperties='Times New Roman', fontsize=18)
        plt.xlabel('Predicted label', fontproperties='Times New Roman', fontsize=18)
        plt.tight_layout()

        accuracy_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        support = cm.sum(axis=1)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f'{cm[i, j]}\n{accuracy_matrix[i, j]:.2%}\n{support[i]}',
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                    fontproperties='Times New Roman')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close()

    def show_roc_curve(self, fpr, tpr, roc_auc, filename=None):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.title('ROC Curve')
        #plt.legend(loc="lower right")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
        plt.close()

def train(args):

    model = SpectrumModel(in_channel=args.in_channel, out_class=args.out_class, dropout=args.dropout)

    print(model)

    full_dataset = SpectrumDataset1D(csv_file=args.dataset, in_channel=args.in_channel, max_length= 2656)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger = TensorBoardLogger("logs", name="spectrum_experiment")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_model_{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(logger=logger, 
                    max_epochs=args.epochs, 
                    gpus=1 if torch.cuda.is_available() else 0,
                    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
                    log_every_n_steps=10) 


    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    final_model = SpectrumModel.load_from_checkpoint(best_model_path)
    print(best_model_path)

    test_results = trainer.test(model = final_model, dataloaders=test_loader)[0]
  
    test_accuracy = test_results["test_accuracy"]
    test_precision = test_results["test_precision"]
    test_recall = test_results["test_recall"]
    test_f1 = test_results["test_f1"]

    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)
                    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default= r'D:\Anaconda3\pythonwork\ramanspec_file\data_training\cnn\rec_spectra0519.csv')
    parser.add_argument("--in-channel", type=int, default= 1 )
    parser.add_argument("--out-class", type=int, default= 15 )
    parser.add_argument("--dropout", type=float, default= 0.10 )
    parser.add_argument("--batch-size", type=int, default= 256 )
    parser.add_argument("--num-workers", type=int, default= 2 )
    parser.add_argument("--epochs", type=int, default= 100 )

    args = parser.parse_args()

    train(args)