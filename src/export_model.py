"""
Exporta el mejor modelo entrenado a TorchScript y guarda metadata util.

Uso:
    python src/export_model.py
    python src/export_model.py --config config.yaml --checkpoint checkpoints/phase2_best.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from model import build_M2_finetuned


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str = 'config.yaml') -> dict:
    with open(project_path(path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Exporta el modelo entrenado')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Ruta al archivo de configuracion')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint a exportar; por defecto usa phase2_best.pth')
    parser.add_argument('--output-dir', type=str, default='exports/base_model',
                        help='Carpeta donde se guardan los artefactos exportados')
    parser.add_argument('--filename', type=str, default='fashion_detector_base.ts',
                        help='Nombre del archivo TorchScript')
    return parser.parse_args()


def load_category_names(data_root: Path, annotation_dir: str) -> list[str]:
    cloth_path = data_root / annotation_dir / 'list_category_cloth.txt'
    df = pd.read_csv(
        cloth_path,
        skiprows=2,
        sep=r'\s+',
        names=['category_name', 'category_type'],
    )
    return df['category_name'].tolist()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = project_path(
        args.checkpoint or f"{cfg.get('checkpoint_dir', 'checkpoints/')}/phase2_best.pth"
    )
    data_root = project_path(cfg['data_root'])
    annotation_dir = cfg.get('annotation_dir', 'Anno_coarse')
    category_names = load_category_names(data_root, annotation_dir)

    device = torch.device('cpu')
    model = build_M2_finetuned(cfg['num_classes'], freeze_phase=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    example = torch.randn(1, 3, 224, 224, device=device)
    scripted_model = torch.jit.trace(model, example)

    model_path = output_dir / args.filename
    scripted_model.save(str(model_path))

    metadata = {
        'model_format': 'torchscript',
        'source_checkpoint': str(checkpoint_path),
        'exported_model': str(model_path),
        'num_classes': cfg['num_classes'],
        'class_names': category_names,
        'input_shape': [1, 3, 224, 224],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'best_epoch': ckpt.get('epoch'),
        'best_val_acc': ckpt.get('val_acc'),
        'config_path': str(project_path(args.config)),
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Checkpoint fuente: {checkpoint_path}")
    print(f"Modelo exportado: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Best epoch: {ckpt.get('epoch')} | val_acc: {ckpt.get('val_acc'):.4f}")


if __name__ == '__main__':
    main()
