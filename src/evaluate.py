"""
Evaluación completa del modelo sobre el split de test.
Genera: top-1 / top-5 accuracy, classification report y matriz de confusión.
"""

import os
import sys
import argparse
from pathlib import Path
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path='config.yaml'):
    with open(project_path(path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluacion del modelo de Fashion Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ruta al archivo de configuracion',
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    top1_correct = top5_correct = total = 0

    for imgs, labels in tqdm(loader, desc='Evaluando'):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
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
    ax.set_title('Matriz de Confusion - DeepFashion', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matriz guardada en {save_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Config: {project_path(args.config)}")

    _, val_tf = get_transforms()
    annotation_dir = cfg.get('annotation_dir', 'Anno_coarse')
    test_ds = DeepFashionDataset(str(project_path(cfg['data_root'])), 'test', val_tf, annotation_dir)
    num_workers = int(cfg['num_workers'])
    test_dl = DataLoader(test_ds, batch_size=cfg['batch_size'],
                         shuffle=False, num_workers=num_workers,
                         pin_memory=DEVICE.type == 'cuda',
                         persistent_workers=num_workers > 0)

    # Cargar modelo
    model = build_M2_finetuned(cfg['num_classes'], freeze_phase=False).to(DEVICE)
    checkpoint_dir = project_path(cfg.get('checkpoint_dir', 'checkpoints/'))
    ckpt  = torch.load(checkpoint_dir / 'phase2_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Modelo cargado (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    top1, top5, preds, labels = evaluate(model, test_dl, cfg['num_classes'])
    print(f"\n{'='*40}")
    print(f"Top-1 Accuracy : {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 Accuracy : {top5:.4f} ({top5*100:.2f}%)")
    print(f"{'='*40}\n")

    # Reporte por clase
    class_names = test_ds.category_names
    report = classification_report(
        labels,
        preds,
        labels=list(range(cfg['num_classes'])),
        target_names=class_names,
        zero_division=0,
    )
    print(report)

    log_dir = project_path(cfg.get('log_dir', 'logs/'))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}\n\n")
        f.write(report)

    # Matriz de confusión
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, log_dir / 'confusion_matrix.png')


if __name__ == '__main__':
    main()
