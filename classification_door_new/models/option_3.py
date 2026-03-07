import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score

from classification_door_new.models.datasets import MyDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    OUTPUT_DIR = r"D:\Metro_Safety_Monitor\classification_door_new\ket_qua_option3"        
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("suwr dung",device)
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    
    num_feats = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_feats, 2)
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3,0.3,0.3),
        transforms.RandomPerspective(0.2),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


    train_ds = MyDataset(root_dir=r"D:\Metro_Safety_Monitor\classification_door_new\data\train",transform = train_transform)
    train_dataloader = DataLoader(
            dataset=train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last= False
        )
    test_dataset = MyDataset(
    root_dir=r"D:\Metro_Safety_Monitor\classification_door_new\data\test",
    transform=test_transform,
    class_to_idx=train_ds.class_to_idx
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_p': [], 'val_r': [], 'val_f1': []}
    best_acc = 0.0
    EPOCH = 50
    best_val_loss = float("inf")
    patience = 2
    counter = 0
    writer = SummaryWriter(log_dir=LOG_DIR)
    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        v_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_t_loss = running_loss / len(train_dataloader)
        avg_v_loss = v_loss / len(test_dataloader)
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = 100. * (all_preds == all_labels).mean()
        p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        metrics = [avg_t_loss, avg_v_loss, acc, p, r, f1]
        for k, v in zip(history.keys(), metrics):
            history[k].append(v)

        writer.add_scalars('Loss', {'Train': avg_t_loss, 'Val': avg_v_loss}, epoch)
        writer.add_scalar('Accuracy/Val', acc, epoch)
        writer.add_scalar('F1_Score/Val', f1, epoch)

        print(f"Epoch [{epoch}/{EPOCH}] Loss: {avg_v_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model_v1.pth'))

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'last_model_v1.pth'))

        # early stopping
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step()       
    writer.close()

    def save_plot(data, label, filename, title, color):
        plt.figure(figsize=(10, 6))
        plt.plot(data, color=color, linewidth=2, label=label)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
        plt.close()

    save_plot(history['val_acc'], 'Accuracy (%)', 'val_accuracy.png', 'Validation Accuracy', 'blue')
    save_plot(history['val_p'], 'Precision', 'val_precision.png', 'Validation Precision', 'green')
    save_plot(history['val_r'], 'Recall', 'val_recall.png', 'Validation Recall', 'orange')
    save_plot(history['val_f1'], 'F1-Score', 'val_f1.png', 'Validation F1-Score', 'red')

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='teal')
    plt.plot(history['val_loss'], label='Val Loss', color='magenta')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(FIG_DIR, 'loss_curves.png'), dpi=300)
    plt.close()

    print(f"Đã lưu kết quả")