# Config
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_densenet121_opw(data_dir, num_workers=4):
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])

    # Load datasets
    image_datasets = {
        phase: datasets.ImageFolder(os.path.join(data_dir, phase), transform=transform)
        for phase in ['train', 'val', 'test']
    }

    dataloaders = {
        phase: DataLoader(image_datasets[phase], batch_size=batch_size, shuffle=(phase == 'train'),
                           num_workers=num_workers)
        for phase in ['train', 'val', 'test']
    }

    # Model
    model = models.densenet121(weights="DEFAULT")
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 1))
    
    # Froze all the layers to only evaluate with the pretained weights
    for param in model.parameters():
        param.requires_grad = False


    model = model.to(device)

    # Only evaluation, no training
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Test'):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())

    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds)
    test_rec = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)

    print(f"\nTest Results:\nAccuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")