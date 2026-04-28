"""
Evaluación completa del modelo sobre el split de test.
Genera: top-1 / top-5 accuracy, classification report y matriz de confusión.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import DeepFashionDataset, get_transforms
from model import build_M2_finetuned

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(path='../config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    top1_correct = top5_correct = total = 0

    for imgs, labels in tqdm(loader, desc='Evaluando'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)

        # Top-1
        top1 = outputs.argmax(1)
        top1_correct += (top1 == labels).sum().item()

        # Top-5
        top5 = outputs.topk(5, dim=1).indices
        for i, lbl in enumerate(labels):
            if lbl in top5[i]:
                top5_correct += 1

        total += labels.size(0)
        all_preds.extend(top1.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    return top1_acc, top5_acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path='../logs/confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicho', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusión — DeepFashion', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matriz guardada en {save_path}")


def main():
    cfg = load_config()

    _, val_tf = get_transforms()
    test_ds = DeepFashionDataset(cfg['data_root'], 'test', val_tf)
    test_dl = DataLoader(test_ds, batch_size=cfg['batch_size'],
                         shuffle=False, num_workers=cfg['num_workers'])

    # Cargar modelo
    model = build_M2_finetuned(cfg['num_classes'], freeze_phase=False).to(DEVICE)
    ckpt  = torch.load('../checkpoints/phase2_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Modelo cargado (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    top1, top5, preds, labels = evaluate(model, test_dl, cfg['num_classes'])
    print(f"\n{'='*40}")
    print(f"Top-1 Accuracy : {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 Accuracy : {top5:.4f} ({top5*100:.2f}%)")
    print(f"{'='*40}\n")

    # Reporte por clase
    class_names = test_ds.category_names
    report = classification_report(labels, preds,
                                   target_names=class_names, zero_division=0)
    print(report)

    os.makedirs('../logs', exist_ok=True)
    with open('../logs/classification_report.txt', 'w') as f:
        f.write(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}\n\n")
        f.write(report)

    # Matriz de confusión
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names)


if __name__ == '__main__':
    main()
