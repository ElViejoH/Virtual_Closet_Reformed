# 👗 Fashion Detector — ResNet50 + DeepFashion

Clasificación de prendas de moda usando ResNet50 con fine-tuning sobre el dataset **DeepFashion** (Category and Attribute Prediction Benchmark).

---

## 📁 Estructura

```
fashion-detector/
├── data/
│   └── raw/                  ← Dataset DeepFashion aquí
├── src/
│   ├── model.py              ← Arquitectura ResNet50 fine-tuned
│   ├── dataset.py            ← DeepFashionDataset + transforms
│   ├── train.py              ← Entrenamiento en 2 fases
│   ├── evaluate.py           ← Métricas + matriz de confusión
│   └── predict.py            ← Inferencia por imagen o carpeta
├── checkpoints/              ← Pesos guardados automáticamente
├── logs/                     ← TensorBoard logs
├── config.yaml               ← Todos los hiperparámetros
└── requirements.txt
```

---

## ⚙️ Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 📦 Dataset

1. Solicita acceso en: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
2. Descarga el **Category and Attribute Prediction Benchmark**
3. Extrae en `data/raw/` con esta estructura:

```
data/raw/
├── img/
├── Anno/
│   ├── list_category_img.txt
│   └── list_category_cloth.txt
└── Eval/
    └── list_eval_partition.txt
```

---

## 🏋️ Entrenamiento

```bash
# Desde la raíz del proyecto
python src/train.py
```

El proceso se divide en dos fases automáticamente:

| Fase | Capas activas | Optimizer | Epochs | LR |
|------|--------------|-----------|--------|----|
| 1    | Solo `fc`    | Adam      | 5      | 1e-3 |
| 2    | `layer3` + `layer4` + `fc` | SGD + Cosine | 20 | 1e-4 |

Monitoreo en tiempo real:
```bash
tensorboard --logdir logs/
```

---

## 📊 Evaluación

```bash
python src/evaluate.py
```

Genera en `logs/`:
- `classification_report.txt` — Precision, Recall, F1 por clase
- `confusion_matrix.png` — Matriz de confusión visual

---

## 🔍 Inferencia

```bash
# Una imagen
python src/predict.py --image path/imagen.jpg --topk 5

# Carpeta completa
python src/predict.py --folder path/carpeta/
```

---

## 🏗️ Modelo

```python
ResNet50 (pretrained ImageNet)
└── fc → Dropout(0.3) → Linear(2048, 512) → ReLU → Linear(512, 50)
```

- **50 categorías** de DeepFashion
- Fine-tuning en 2 fases para evitar catastrofic forgetting
- Label smoothing 0.1 + Cosine Annealing en fase 2
- Mixed precision (AMP) si hay GPU disponible

---

## 📋 Checklist

- [ ] Descargar DeepFashion y colocar en `data/raw/`
- [ ] `pip install -r requirements.txt`
- [ ] Verificar DataLoader: `python src/dataset.py`
- [ ] Entrenamiento: `python src/train.py`
- [ ] Evaluación: `python src/evaluate.py`
- [ ] Inferencia: `python src/predict.py --image ...`
