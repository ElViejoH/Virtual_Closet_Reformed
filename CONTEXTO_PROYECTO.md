# Contexto del Proyecto: Fashion Detector

## 1. Resumen general

Este proyecto implementa un clasificador de prendas de vestir basado en `ResNet50` y entrenado sobre el benchmark `DeepFashion Category and Attribute Prediction`.

El objetivo principal es reconocer una prenda dentro de 50 categorias usando fine-tuning sobre un modelo preentrenado en ImageNet. Ademas del entrenamiento base, el proyecto evoluciono para:

- adaptarse a la estructura real descargada del dataset (`Anno_coarse`);
- ejecutar entrenamiento en 2 fases;
- probar una estrategia de rebalanceo con `WeightedRandomSampler`;
- evaluar el modelo con metricas detalladas;
- exportar el modelo entrenado a `TorchScript`;
- versionar artefactos pesados en GitHub usando `Git LFS`.

## 2. Problema que resuelve

El sistema toma una imagen de una prenda y predice su categoria entre clases como `Dress`, `Tee`, `Blouse`, `Jacket`, `Skirt`, `Jeans`, `Romper` y otras del benchmark de DeepFashion.

Tambien permite:

- inferencia sobre una sola imagen;
- inferencia por lotes sobre una carpeta;
- evaluacion completa sobre el split de test;
- exportacion del modelo para despliegue o integracion posterior.

## 3. Stack tecnico

- `Python`
- `PyTorch`
- `torchvision`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tensorboard`
- `PyYAML`

El proyecto fue probado en Windows con GPU `NVIDIA GeForce RTX 4060` y PyTorch con CUDA.

## 4. Estructura funcional del repositorio

- `src/dataset.py`: carga el dataset DeepFashion, une splits y etiquetas, aplica transforms y expone nombres de clase.
- `src/model.py`: define la arquitectura basada en `ResNet50`.
- `src/train.py`: entrenamiento en dos fases, con soporte para sampler ponderado, class weights, AMP y TensorBoard.
- `src/evaluate.py`: calcula `Top-1`, `Top-5`, classification report y matriz de confusion.
- `src/predict.py`: inferencia por imagen o carpeta.
- `src/export_model.py`: exporta el mejor checkpoint a `TorchScript` y genera metadata.
- `config.yaml`: configuracion base del entrenamiento principal.
- `config_sampler_only.yaml`: configuracion alternativa con `WeightedRandomSampler`.
- `checkpoints_*`: pesos entrenados.
- `logs_*`: reportes y eventos de TensorBoard.
- `exports/base_model/`: artefactos exportados del modelo base.

## 5. Dataset y adaptacion realizada

El proyecto trabaja con `DeepFashion Category and Attribute Prediction Benchmark`.

Durante el desarrollo se ajusto el pipeline para la estructura real disponible en la descarga local:

- se usa `data/raw/Anno_coarse/` en lugar de `Anno/`;
- los archivos de anotacion y particion se leen con `skiprows=2`;
- las etiquetas se convierten de indices `1-based` a `0-based`;
- se mantienen `50` categorias.

La estructura esperada es:

```text
data/raw/
├── img/
├── Anno_coarse/
│   ├── list_category_img.txt
│   └── list_category_cloth.txt
└── Eval/
    └── list_eval_partition.txt
```

## 6. Arquitectura del modelo

El modelo parte de `ResNet50(weights=IMAGENET1K_V2)` y reemplaza la capa final por:

```text
Dropout(0.3) -> Linear(2048, 512) -> ReLU -> Linear(512, 50)
```

La logica de entrenamiento distingue dos etapas:

1. Fase 1:
   se congela el backbone y solo se entrena el clasificador final.
2. Fase 2:
   se descongelan `layer3`, `layer4` y `fc`, mientras las capas iniciales quedan congeladas.

## 7. Flujo de entrenamiento implementado

El script `src/train.py` soporta configuracion por archivo YAML y ejecuta automaticamente:

1. carga del dataset de entrenamiento y validacion;
2. calculo opcional de pesos por clase;
3. uso opcional de `WeightedRandomSampler`;
4. entrenamiento por fases;
5. guardado del mejor checkpoint por fase;
6. logging en TensorBoard.

Detalles relevantes del pipeline:

- `batch_size: 64`
- `num_workers: 2`
- `label_smoothing: 0.1`
- AMP en GPU con `torch.amp.autocast`
- `Adam` en fase 1
- `SGD + momentum + CosineAnnealingLR` en fase 2
- `pin_memory` y `persistent_workers` cuando aplica

## 8. Variantes desarrolladas

### 8.1 Modelo base

Configuracion en `config.yaml`:

- `use_weighted_sampler: false`
- `use_class_weights: false`
- checkpoints en `checkpoints/`
- logs en `logs/`

Esta version prioriza accuracy global y fue la base usada para exportacion.

### 8.2 Variante con sampler ponderado

Configuracion en `config_sampler_only.yaml`:

- `use_weighted_sampler: true`
- `use_class_weights: false`
- checkpoints en `checkpoints_sampler_only/`
- logs en `logs_sampler_only/`

Esta version busca compensar el desbalance de clases sobre-muestreando clases escasas.

## 9. Resultados obtenidos

### 9.1 Modelo base exportado

Segun `exports/base_model/classification_report.txt`:

- `Top-1: 0.6443`
- `Top-5: 0.9168`
- `accuracy global test: 0.64`
- `weighted avg f1-score: 0.62`

Segun `exports/base_model/metadata.json`:

- `best_epoch: 23`
- `best_val_acc: 0.650225`
- formato exportado: `TorchScript`

Clases con mejor comportamiento relativo en esta corrida:

- `Dress`
- `Tee`
- `Jeans`
- `Shorts`
- `Skirt`
- `Leggings`
- `Romper`

Se observa desempeno bajo en clases muy raras o con poco soporte, por ejemplo:

- `Halter`
- `Peacoat`
- `Capris`
- `Gauchos`
- `Kaftan`
- `Robe`

### 9.2 Variante con sampler ponderado

Segun `logs_sampler_only/classification_report.txt`:

- `Top-1: 0.4745`
- `Top-5: 0.8513`
- `accuracy global test: 0.47`
- `weighted avg f1-score: 0.50`

Interpretacion:

- mejora recall en varias clases minoritarias;
- reduce de forma importante la precision/accuracy global;
- sirve como experimento de rebalanceo, pero no supera al modelo base como version principal.

## 10. Evaluacion e inferencia

### Evaluacion

`src/evaluate.py` carga el mejor `phase2_best.pth` de la configuracion indicada y genera:

- `classification_report.txt`
- `confusion_matrix.png`

Tambien reporta:

- `Top-1 Accuracy`
- `Top-5 Accuracy`

### Inferencia

`src/predict.py` permite:

- `--image` para una sola imagen;
- `--folder` para procesar una carpeta completa;
- `--topk` para ajustar la cantidad de clases reportadas.

La inferencia usa:

- resize a `224x224`;
- normalizacion con estadisticas de ImageNet;
- `softmax` para obtener probabilidades.

## 11. Exportacion del modelo

Se agrego `src/export_model.py` para exportar el modelo entrenado a `TorchScript`.

Salida principal:

- `exports/base_model/fashion_detector_base.ts`
- `exports/base_model/metadata.json`

La metadata incluye:

- ruta del checkpoint fuente;
- ruta del modelo exportado;
- numero de clases;
- nombres de clases;
- shape de entrada;
- normalizacion;
- mejor epoca y mejor `val_acc`.

Esto deja el proyecto listo para una etapa posterior de integracion con otra aplicacion o servicio.

## 12. Artefactos importantes generados

- `checkpoints/phase1_best.pth`
- `checkpoints/phase2_best.pth`
- `checkpoints_sampler_only/phase1_best.pth`
- `checkpoints_sampler_only/phase2_best.pth`
- `exports/base_model/fashion_detector_base.ts`
- `exports/base_model/metadata.json`
- `exports/base_model/classification_report.txt`
- `exports/base_model/confusion_matrix.png`
- `logs_sampler_only/classification_report.txt`
- `logs_sampler_only/confusion_matrix.png`

## 13. Historial de desarrollo reflejado en Git

Commits relevantes en la historia reciente:

- `d419c1e`: proyecto inicial DeepFashion con ResNet50.
- `7d53996`: ajuste menor del README.
- `611291e`: incorporacion del flujo ampliado de entrenamiento, exportacion y resultados.
- `5e09d71`: configuracion de `Git LFS` para artefactos pesados.

## 14. GitHub y manejo de archivos pesados

Durante la subida a GitHub se presento un bloqueo por archivos mayores a `100 MB`, especialmente los checkpoints `.pth`.

Para resolverlo se hizo lo siguiente:

- configuracion de `Git LFS`;
- tracking de `*.pth`;
- tracking de `exports/base_model/fashion_detector_base.ts`;
- reescritura local de la historia no publicada para migrar esos binarios a LFS;
- publicacion final en `origin/main`.

Esto es importante porque cualquier clon del repositorio necesita `git lfs` para descargar correctamente esos artefactos pesados.

## 15. Estado actual del proyecto

El proyecto queda en un estado funcional y reproducible, con:

- entrenamiento configurable;
- evaluacion automatizada;
- inferencia local;
- exportacion del modelo;
- comparacion entre una variante base y una variante con rebalanceo;
- artefactos versionados en GitHub.

La version base es, por ahora, la candidata mas fuerte para uso principal porque mantiene el mejor balance entre accuracy global y utilidad practica.

## 16. Posibles siguientes pasos

- probar `class_weights` sin sampler para comparar contra ambas variantes;
- guardar un resumen tabular comparativo entre experimentos;
- separar mejor los artefactos de entrenamiento y los de despliegue;
- agregar un script o notebook de demo visual;
- construir una API o interfaz para consumir el modelo exportado.
