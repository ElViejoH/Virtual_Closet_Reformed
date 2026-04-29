"""
Entrenamiento en dos fases para ResNet50 fine-tuned sobre DeepFashion.

Fase 1 (freeze_phase=True)  -> Solo entrena el clasificador fc  (~5 epochs)
Fase 2 (freeze_phase=False) -> Fine-tuning de layer3, layer4 y fc (~20 epochs)
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Permite importar desde src/
sys.path.insert(0, os.path.dirname(__file__))
from dataset import DeepFashionDataset, get_transforms
from model import build_M2_finetuned


# ─── Configuración ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str = 'config.yaml') -> dict:
    with open(project_path(path), encoding='utf-8') as f:
        return yaml.safe_load(f)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


def compute_class_weights(train_ds, num_classes: int) -> torch.Tensor:
    """Calcula pesos inversamente proporcionales a la frecuencia por clase."""
    counts = torch.bincount(
        torch.as_tensor(train_ds.data['category_label'].to_numpy(), dtype=torch.long),
        minlength=num_classes,
    ).float()

    nonzero = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32)
    weights[nonzero] = counts[nonzero].sum() / (num_classes * counts[nonzero])
    return weights


def build_train_sampler(train_ds, class_weights: torch.Tensor):
    """Crea un sampler ponderado para sobre-muestrear clases escasas."""
    sample_weights = class_weights[
        torch.as_tensor(train_ds.data['category_label'].to_numpy(), dtype=torch.long)
    ].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─── Loops de entrenamiento / validación ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='  Train', leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:  # Mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='  Val  ', leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ─── Entrenamiento por fases ──────────────────────────────────────────────────

def run_phase(phase: int, model, train_dl, val_dl,
              epochs: int, lr: float, writer: SummaryWriter,
              epoch_offset: int = 0, cfg: dict = None,
              class_weights: torch.Tensor | None = None):
    """Ejecuta una fase completa de entrenamiento."""

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1,
        weight=class_weights.to(DEVICE) if class_weights is not None else None,
    )
    checkpoint_dir = project_path(cfg.get('checkpoint_dir', 'checkpoints/'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if phase == 1:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        scheduler = None
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    best_acc = 0.0

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, scaler)
        vl_loss, vl_acc = validate(model, val_dl, criterion)

        if scheduler:
            scheduler.step()

        global_epoch = epoch_offset + epoch + 1
        writer.add_scalars('Loss', {'train': tr_loss, 'val': vl_loss}, global_epoch)
        writer.add_scalars('Acc',  {'train': tr_acc,  'val': vl_acc},  global_epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_epoch)

        print(f"[Fase {phase} | Epoch {epoch+1:02d}/{epochs}] "
              f"Loss {tr_loss:.4f} -> {vl_loss:.4f} | "
              f"Acc  {tr_acc:.4f} -> {vl_acc:.4f}")

        # Guardar mejor modelo
        if vl_acc > best_acc:
            best_acc = vl_acc
            ckpt_path = checkpoint_dir / f'phase{phase}_best.pth'
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': vl_acc,
            }, ckpt_path)
            print(f"  Nuevo mejor modelo guardado (val_acc={vl_acc:.4f})")

    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Entrenamiento de Fashion Detector')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ruta al archivo de configuracion',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Config: {project_path(args.config)}")
    print(f"Dispositivo: {DEVICE}")

    # DataLoaders
    train_tf, val_tf = get_transforms()
    annotation_dir = cfg.get('annotation_dir', 'Anno_coarse')
    data_root = str(project_path(cfg['data_root']))
    num_workers = int(cfg['num_workers'])
    num_classes = int(cfg['num_classes'])
    train_ds = DeepFashionDataset(data_root, 'train', train_tf, annotation_dir)
    val_ds   = DeepFashionDataset(data_root, 'val',   val_tf,   annotation_dir)
    class_weights = compute_class_weights(train_ds, num_classes)
    use_weighted_sampler = bool(cfg.get('use_weighted_sampler', False))
    use_class_weights = bool(cfg.get('use_class_weights', False))

    if use_weighted_sampler:
        train_sampler = build_train_sampler(train_ds, class_weights)
        print("Sampler de entrenamiento: WeightedRandomSampler")
    else:
        train_sampler = None
        print("Sampler de entrenamiento: shuffle aleatorio")

    if use_class_weights:
        nonzero_weights = class_weights[class_weights > 0]
        print(
            "Loss ponderada activada: "
            f"min={nonzero_weights.min():.4f}, max={nonzero_weights.max():.4f}"
        )
    else:
        print("Loss ponderada: desactivada")

    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'],
                          shuffle=train_sampler is None,
                          sampler=train_sampler,
                          num_workers=num_workers,
                          pin_memory=DEVICE.type == 'cuda',
                          persistent_workers=num_workers > 0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=num_workers,
                          pin_memory=DEVICE.type == 'cuda',
                          persistent_workers=num_workers > 0)

    writer = SummaryWriter(str(project_path(cfg.get('log_dir', 'logs/'))))
    active_class_weights = class_weights if use_class_weights else None

    # ── Fase 1: solo clasificador ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("FASE 1: Entrenamiento del clasificador (backbone congelado)")
    print("="*60)
    model = build_M2_finetuned(num_classes, freeze_phase=True).to(DEVICE)
    model = run_phase(1, model, train_dl, val_dl,
                      epochs=cfg['phase1_epochs'], lr=cfg['phase1_lr'],
                      writer=writer, epoch_offset=0, cfg=cfg,
                      class_weights=active_class_weights)

    # ── Fase 2: fine-tuning profundo ──────────────────────────────────────────
    print("\n" + "="*60)
    print("FASE 2: Fine-tuning de layer3, layer4 y fc")
    print("="*60)
    model_ft = build_M2_finetuned(num_classes, freeze_phase=False).to(DEVICE)

    # Cargar pesos de fase 1
    ckpt = torch.load(project_path(cfg.get('checkpoint_dir', 'checkpoints/')) / 'phase1_best.pth',
                      map_location=DEVICE)
    model_ft.load_state_dict(ckpt['model_state_dict'])
    print("  Pesos de Fase 1 cargados correctamente.")

    model_ft = run_phase(2, model_ft, train_dl, val_dl,
                         epochs=cfg['phase2_epochs'], lr=cfg['phase2_lr'],
                         writer=writer, epoch_offset=cfg['phase1_epochs'], cfg=cfg,
                         class_weights=active_class_weights)

    writer.close()
    print("\nEntrenamiento completo. Mejor modelo en checkpoints/phase2_best.pth")


if __name__ == '__main__':
    main()
