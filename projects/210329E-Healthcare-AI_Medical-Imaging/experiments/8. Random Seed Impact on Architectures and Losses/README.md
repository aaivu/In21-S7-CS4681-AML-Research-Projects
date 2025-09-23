
## Ensemble Model Selection Based on Random Seed and Architecture Diversity

To maximize the diversity and robustness of the proposed ensemble, I analyzed the impact of random seed variation and model architecture on performance. Models were selected based on their average AUROC scores, per-class performance, and architectural diversity.

**Selected models for the future ensemble:**

| Model Name                                 | Architecture         | Loss        | Seed | Avg AUROC | Avg F1  | Loss    |
|---------------------------------------------|----------------------|-------------|------|-----------|---------|---------|
| DenseNet-121 (Focal Loss, seed 22)          | DenseNet-121         | Focal Loss  | 22   | 0.8475    | 0.3852  | 0.0421  |
| DenseNet-121 (Focal Loss, seed 32)          | DenseNet-121         | Focal Loss  | 32   | 0.8458    | 0.3679  | 0.0422  |
| DenseNet-121 (Focal Loss, seed 42)          | DenseNet-121         | Focal Loss  | 42   | 0.8514    | 0.3803  | 0.0415  |
| DenseNet-121 (ZLPR Loss, seed 32)           | DenseNet-121         | ZLPR Loss   | 32   | 0.8479    | 0.3762  | 1.6263  |
| DenseNet-121 (ZLPR Loss, seed 42)           | DenseNet-121         | ZLPR Loss   | 42   | 0.8462    | 0.3621  | 1.6268  |
| DenseNet-121 + CBAM (Focal Loss, seed 42)   | DenseNet-121 + CBAM  | Focal Loss  | 42   | 0.8480    | 0.3787  | 0.0419  |
| EfficientNet-B2 (Focal Loss, seed 42)       | EfficientNet-B2      | Focal Loss  | 42   | 0.8482    | 0.3676  | 0.0423  |

### Per-Class AUROC for Selected Models

| Class                | D121-Focal-22 | D121-Focal-32 | D121-Focal-42 | D121-ZLPR-32 | D121-ZLPR-42 | D121+CBAM-Focal-42 | EffB2-Focal-42 |
|----------------------|:-------------:|:-------------:|:-------------:|:------------:|:------------:|:------------------:|:--------------:|
| Atelectasis          | 0.8129        | 0.8187        | 0.8146        | 0.8153       | 0.8030       | 0.8138             | 0.8168         |
| Cardiomegaly         | 0.9388        | 0.9406        | 0.9325        | 0.9244       | 0.9249       | 0.9364             | 0.9259         |
| Consolidation        | 0.7772        | 0.7766        | 0.7871        | 0.7793       | 0.7764       | 0.7774             | 0.7799         |
| Edema                | 0.8992        | 0.8806        | 0.8841        | 0.8928       | 0.8957       | 0.8950             | 0.9064         |
| Effusion             | 0.8993        | 0.9014        | 0.9015        | 0.9024       | 0.8990       | 0.8991             | 0.9032         |
| Emphysema            | 0.9681        | 0.9669        | 0.9656        | 0.9614       | 0.9655       | 0.9606             | 0.9612         |
| Fibrosis             | 0.8441        | 0.8185        | 0.8207        | 0.8507       | 0.8564       | 0.8379             | 0.8177         |
| Hernia               | 0.9801        | 0.9783        | 0.9936        | 0.9960       | 0.9830       | 0.9973             | 0.9733         |
| Infiltration         | 0.6970        | 0.6983        | 0.7044        | 0.7070       | 0.6983       | 0.7058             | 0.7093         |
| Mass                 | 0.8983        | 0.9029        | 0.9122        | 0.8969       | 0.9068       | 0.9077             | 0.8974         |
| Nodule               | 0.7758        | 0.7653        | 0.7780        | 0.7705       | 0.7671       | 0.7648             | 0.7848         |
| Pleural_Thickening   | 0.8002        | 0.7932        | 0.8124        | 0.8009       | 0.8015       | 0.7932             | 0.7845         |
| Pneumonia            | 0.6974        | 0.7243        | 0.7229        | 0.6955       | 0.6817       | 0.7054             | 0.7338         |
| Pneumothorax         | 0.8761        | 0.8757        | 0.8902        | 0.8776       | 0.8873       | 0.8781             | 0.8808         |

> **Note:**
> Model names follow the convention:  D121 = DenseNet-121, EffB2 = EfficientNet-B2, CBAM = Attention module, Focal/ZLPR = Loss function, Number = random seed

These models were chosen to ensure both high individual performance and diversity in architecture and initialization, which is expected to improve the effectiveness of the future ensemble.
