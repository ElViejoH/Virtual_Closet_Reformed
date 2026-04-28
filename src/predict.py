"""
Inferencia sobre imágenes individuales o directorios completos.

Uso:
    python predict.py --image ruta/imagen.jpg
    python predict.py --folder ruta/carpeta/
"""

import os
import sys
import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import build_M2_finetuned

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INFERENCE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_config(path='../config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_category_names(data_root: str):
    """Carga los nombres de categorías desde DeepFashion."""
    import pandas as pd
    cloth_path = os.path.join(data_root, 'Anno', 'list_category_cloth.txt')
    df = pd.read_csv(cloth_path, skiprows=1, sep=r'\s+',
                     names=['category_name', 'category_type'])
    return df['category_name'].tolist()


def load_model(cfg):
    model = build_M2_finetuned(cfg['num_classes'], freeze_phase=False)
    ckpt  = torch.load('../checkpoints/phase2_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def predict_image(model, image_path: str, category_names: list, top_k: int = 5):
    """
    Predice la categoría de una imagen.

    Returns:
        list[tuple]: [(nombre_clase, probabilidad), ...]  top_k resultados
    """
    img = Image.open(image_path).convert('RGB')
    tensor = INFERENCE_TF(img).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze(0)
    topk   = probs.topk(top_k)

    results = [
        (category_names[idx.item()], prob.item())
        for prob, idx in zip(topk.values, topk.indices)
    ]
    return results


def predict_folder(model, folder_path: str, category_names: list):
    """Predice todas las imágenes en una carpeta."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in extensions]

    print(f"\nProcesando {len(files)} imágenes en '{folder_path}'...")
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        results = predict_image(model, fpath, category_names, top_k=3)
        top_name, top_prob = results[0]
        print(f"  {fname:40s} → {top_name} ({top_prob:.2%})")


def main():
    parser = argparse.ArgumentParser(description='Inferencia de prendas de moda')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image',  type=str, help='Ruta a una imagen')
    group.add_argument('--folder', type=str, help='Ruta a una carpeta de imágenes')
    parser.add_argument('--topk',  type=int, default=5, help='Número de predicciones top-k')
    args = parser.parse_args()

    cfg = load_config()
    category_names = load_category_names(cfg['data_root'])
    model = load_model(cfg)

    if args.image:
        print(f"\nImagen: {args.image}")
        print("-" * 50)
        results = predict_image(model, args.image, category_names, top_k=args.topk)
        for rank, (name, prob) in enumerate(results, 1):
            bar = '█' * int(prob * 30)
            print(f"  #{rank}  {name:30s}  {bar:<30s}  {prob:.2%}")

    elif args.folder:
        predict_folder(model, args.folder, category_names)


if __name__ == '__main__':
    main()
