import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import copy


class MammoCNN(nn.Module):
    def __init__(self):
        super(MammoCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Ajusta al tamaño fijo

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = self.fc_block(x)
        return x
    
def train_and_eval_mammocnn(data_dir, num_epochs=50, num_workers=4):
    batch_size = 32
    learning_rate = 1e-4
    patience = 10  # Early stopping patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.1988], [0.2481])
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
    model = MammoCNN()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Early stopping vars
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}\n' + '-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()}'):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).int()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds)
            rec = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}')

            # Early stopping check
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print("- Validation loss improved. Saving model...")

                    # Save the best model
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    save_dir = os.path.join(script_dir, 'checkpoint')
                    os.makedirs(save_dir, exist_ok=True)

                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_mammocnn.pth'))
                else:
                    patience_counter += 1
                    print(f"- No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        break


    # Evaluation on test set
    # Load best model 
    model.load_state_dict(best_model_wts)
    print("\nEvaluating on test set...\n" + "-"*30)
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