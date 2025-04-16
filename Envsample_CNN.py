import torch
from torchvision import transforms
from training_CNN1D import SpectrumModel as SpectrumModel1D
from training_CNN2D import SpectrumModel as SpectrumModel2D
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import SpectrumDataset2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd


def load_dataset(data_path):
    data = pd.read_csv(data_path)
    features = data.columns[1:]
    X = data[features].values
    y = data.iloc[:, 0].values
    return X, y

def prepare_data(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

def evaluate_model_1D(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            output = model(X_batch)
            pred = torch.argmax(output, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm, y_true, y_pred

def evaluate_model_2D(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm, y_true, y_pred


model_type = "2D"

datapath_1D = r"D:\Anaconda3\pythonwork\ramanspec_file\data\rec250210_Envsample_nofluoRaman.csv"
datapath_2D = r"D:\Anaconda3\pythonwork\ramanspec_file\data_training\cnn\rec250210_Ramandata_images"
checkpoint_path_1D = r"D:\Anaconda3\pythonwork\ramanspec_file\checkpoints\best_model_epoch=08-val_loss=0.04-v1.ckpt"
checkpoint_path_2D = r"D:\Anaconda3\pythonwork\ramanspec_file\checkpoints_2D\best_model_epoch=10-val_loss=0.04.ckpt"

if model_type == "1D":
    
    X_test, y_test = load_dataset(datapath_1D)

    X_test_tensor, y_test_tensor = prepare_data(X_test, y_test)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SpectrumModel1D.load_from_checkpoint(checkpoint_path_1D)

    accuracy, precision, recall, f1, cm, y_true, y_pred = evaluate_model_1D(model, test_loader)

    print(y_true)
    print(y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted label', fontsize = 16)
    plt.ylabel('True labels', fontsize = 16)
    plt.title('Confusion Matrix (CNN1D)', fontsize = 18)
    plt.show()
    

elif model_type == "2D":
    transform =  transforms.Grayscale()
    batch_size = 64

    dataset = SpectrumDataset2D(datapath_2D, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = SpectrumModel2D.load_from_checkpoint(checkpoint_path_2D)

    accuracy, precision, recall, f1, cm, y_true, y_pred = evaluate_model_2D(model, data_loader)

    print(y_true)
    print(y_pred)

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted labels', fontsize = 16)
    plt.ylabel('True labels', fontsize = 16)
    plt.title('Confusion Matrix (CNN2D)', fontsize = 18)
    plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

