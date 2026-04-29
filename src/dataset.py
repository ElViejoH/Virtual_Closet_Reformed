import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFashionDataset(Dataset):
    """
    Dataset para DeepFashion – Category and Attribute Prediction Benchmark.

    Estructura esperada en root_dir:
        root_dir/
        ├── img/                          # imágenes originales
        ├── Anno_coarse/
        │   ├── list_category_img.txt     # imagen -> indice de categoria
        │   └── list_category_cloth.txt   # indice -> nombre de categoria
        └── Eval/
            └── list_eval_partition.txt   # train / val / test split
    """

    def __init__(self, root_dir: str, partition: str = 'train',
                 transform=None, annotation_dir: str = 'Anno_coarse'):
        """
        Args:
            root_dir  : Ruta raíz del dataset DeepFashion.
            partition : 'train', 'val' o 'test'.
            transform : Transformaciones torchvision.
            annotation_dir : Carpeta de anotaciones ('Anno_coarse' en esta descarga).
        """
        self.root_dir       = root_dir
        self.annotation_dir = annotation_dir
        self.transform      = transform

        # ── Cargar split ──────────────────────────────────────────────────────
        eval_path = os.path.join(root_dir, 'Eval', 'list_eval_partition.txt')
        part_df = pd.read_csv(
            eval_path, skiprows=2, sep=r'\s+',
            names=['image_name', 'evaluation_status']
        )
        part_df = part_df[part_df['evaluation_status'] == partition].reset_index(drop=True)

        # ── Cargar etiquetas de categoría ─────────────────────────────────────
        cat_path = os.path.join(root_dir, annotation_dir, 'list_category_img.txt')
        cat_df = pd.read_csv(
            cat_path, skiprows=2, sep=r'\s+',
            names=['image_name', 'category_label']
        )
        cat_df['category_label'] = pd.to_numeric(cat_df['category_label'])

        # ── Merge y ajuste de índice ──────────────────────────────────────────
        self.data = pd.merge(part_df, cat_df, on='image_name').reset_index(drop=True)
        self.data['category_label'] -= 1  # 1-indexed -> 0-indexed

        # ── Cargar nombres de categorías ──────────────────────────────────────
        cloth_path = os.path.join(root_dir, annotation_dir, 'list_category_cloth.txt')
        cloth_df = pd.read_csv(
            cloth_path, skiprows=2, sep=r'\s+',
            names=['category_name', 'category_type']
        )
        self.category_names = cloth_df['category_name'].tolist()

        print(f"[{partition.upper()} | {annotation_dir}] {len(self.data):,} imágenes | "
              f"{len(self.category_names)} categorías")

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"No se pudo abrir la imagen: {img_path}") from e

        label = int(row['category_label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    # ─────────────────────────────────────────────────────────────────────────

    def get_class_name(self, idx: int) -> str:
        """Devuelve el nombre de la categoría dado su índice."""
        return self.category_names[idx]


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_transforms():
    """
    Retorna transformaciones para entrenamiento y validación/test.
    Usa las medias y desviaciones estándar de ImageNet.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader

    DATA_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    train_tf, val_tf = get_transforms()

    ds = DeepFashionDataset(DATA_ROOT, partition='train', transform=train_tf,
                            annotation_dir='Anno_coarse')
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    imgs, labels = next(iter(dl))
    print(f"Batch shapes -> imgs: {imgs.shape} | labels: {labels.shape}")
    print(f"Ejemplo etiqueta: {labels[0].item()} -> {ds.get_class_name(labels[0].item())}")
