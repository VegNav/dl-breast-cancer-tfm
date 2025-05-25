import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EfficientNet_Custom(nn.Module):
    def __init__(self, dropout_rate, fc1_units, fc2_units, fc3_units):
        super(EfficientNet_Custom, self).__init__()
        self.backbone = models.efficientnet_b3(weights='DEFAULT')
        self.backbone.features[0][0] = nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1, bias=False)

        self.fc1 = nn.Linear(1000, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc_out = nn.Linear(fc3_units, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.dropout(x)
        x = self.elu(self.fc3(x))
        x = self.fc_out(x)
        return x

best_params = {
    'dropout': 0.36170713417061484,
    'lr': 0.0005634665795942972,
    'fc1_units': 768,
    'fc2_units': 128,
    'fc3_units': 32,
    'freeze_proportion': 0.6979547715740493,
    'optimizer': 'AdamW'
}

def train_and_eval_efficientnetB3_optuna(data_dir, num_epochs=50, num_workers=4):
    batch_size = 32
    learning_rate = best_params['lr']
    patience = 10  # Early stopping patience
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

    def set_trainable_layers(model, proportion):
        total = len(model.backbone.features)
        freeze_until = int(total * proportion)
        for i, layer in enumerate(model.backbone.features):
            for param in layer.parameters():
                param.requires_grad = i >= freeze_until


    # Model
    model = EfficientNet_Custom(
        dropout_rate=best_params['dropout'],
        fc1_units=best_params['fc1_units'],
        fc2_units=best_params['fc2_units'],
        fc3_units=best_params['fc3_units']
    )

    set_trainable_layers(model, best_params['freeze_proportion'])

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

                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_efficientnetB3Optuna.pth'))
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
