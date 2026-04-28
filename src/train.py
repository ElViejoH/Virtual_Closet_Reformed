"""
Entrenamiento en dos fases para ResNet50 fine-tuned sobre DeepFashion.

Fase 1 (freeze_phase=True)  → Solo entrena el clasificador fc  (~5 epochs)
Fase 2 (freeze_phase=False) → Fine-tuning de layer3, layer4 y fc (~20 epochs)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Permite importar desde src/
sys.path.insert(0, os.path.dirname(__file__))
from dataset import DeepFashionDataset, get_transforms
from model import build_M2_finetuned


# ─── Configuración ───────────────────────────────────────────────────────────

def load_config(path: str = '../config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Loops de entrenamiento / validación ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='  Train', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
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
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ─── Entrenamiento por fases ──────────────────────────────────────────────────

def run_phase(phase: int, model, train_dl, val_dl,
              epochs: int, lr: float, writer: SummaryWriter,
              epoch_offset: int = 0, cfg: dict = None):
    """Ejecuta una fase completa de entrenamiento."""

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs('../checkpoints', exist_ok=True)

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

    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
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
              f"Loss {tr_loss:.4f} → {vl_loss:.4f} | "
              f"Acc  {tr_acc:.4f} → {vl_acc:.4f}")

        # Guardar mejor modelo
        if vl_acc > best_acc:
            best_acc = vl_acc
            ckpt_path = f'../checkpoints/phase{phase}_best.pth'
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': vl_acc,
            }, ckpt_path)
            print(f"  ✔ Nuevo mejor modelo guardado (val_acc={vl_acc:.4f})")

    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    print(f"Dispositivo: {DEVICE}")

    # DataLoaders
    train_tf, val_tf = get_transforms()
    train_ds = DeepFashionDataset(cfg['data_root'], 'train', train_tf)
    val_ds   = DeepFashionDataset(cfg['data_root'], 'val',   val_tf)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'],
                          shuffle=True,  num_workers=cfg['num_workers'], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    writer = SummaryWriter('../logs')
    num_classes = cfg['num_classes']

    # ── Fase 1: solo clasificador ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("FASE 1: Entrenamiento del clasificador (backbone congelado)")
    print("="*60)
    model = build_M2_finetuned(num_classes, freeze_phase=True).to(DEVICE)
    model = run_phase(1, model, train_dl, val_dl,
                      epochs=cfg['phase1_epochs'], lr=cfg['phase1_lr'],
                      writer=writer, epoch_offset=0, cfg=cfg)

    # ── Fase 2: fine-tuning profundo ──────────────────────────────────────────
    print("\n" + "="*60)
    print("FASE 2: Fine-tuning de layer3, layer4 y fc")
    print("="*60)
    model_ft = build_M2_finetuned(num_classes, freeze_phase=False).to(DEVICE)

    # Cargar pesos de fase 1
    ckpt = torch.load('../checkpoints/phase1_best.pth', map_location=DEVICE)
    model_ft.load_state_dict(ckpt['model_state_dict'])
    print("  Pesos de Fase 1 cargados correctamente.")

    model_ft = run_phase(2, model_ft, train_dl, val_dl,
                         epochs=cfg['phase2_epochs'], lr=cfg['phase2_lr'],
                         writer=writer, epoch_offset=cfg['phase1_epochs'], cfg=cfg)

    writer.close()
    print("\n✅ Entrenamiento completo. Mejor modelo en checkpoints/phase2_best.pth")


if __name__ == '__main__':
    main()
