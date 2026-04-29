import torch.nn as nn
from torchvision import models


def build_M2_finetuned(num_classes, freeze_phase=False):
    """
    Construye ResNet50 con fine-tuning para clasificación de prendas.

    Args:
        num_classes (int): Número de categorías (50 para DeepFashion).
        freeze_phase (bool):
            True  -> Fase 1: congela todo excepto fc (solo entrena clasificador).
            False -> Fase 2: descongela layer3, layer4 y fc.

    Returns:
        nn.Module: Modelo listo para entrenamiento.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_phase:
        # Fase 1: congelar todo el backbone
        for p in model.parameters():
            p.requires_grad = False
    else:
        # Fase 2: congelar solo las capas iniciales
        for name, p in model.named_parameters():
            if any(k in name for k in ['layer1', 'layer2', 'conv1', 'bn1']):
                p.requires_grad = False
            else:
                p.requires_grad = True

    # Reemplazar clasificador final
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)
    )

    return model


if __name__ == '__main__':
    import torch
    model = build_M2_finetuned(50, freeze_phase=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Esperado: [2, 50]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable:,} / {total:,}")
