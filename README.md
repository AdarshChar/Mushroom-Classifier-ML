# Mushroom Classifier
### Edible vs. Poisonous Detection using Fine-Tuned ResNet-50
*Google Colab, PyTorch, HuggingFace Transformers, KaggleHub*

---

## 1. Project Overview

This project fine-tunes Microsoft's ResNet-50 (via HuggingFace Transformers) on a Kaggle mushroom image dataset to classify fungi as edible or poisonous. The pipeline runs entirely in Google Colab and handles data download, preprocessing, augmentation, training with class-weighted loss, best-model checkpointing, and single-image inference.

---

## 2. Requirements

### Python Packages
Install all dependencies by running Cell 1 and Cell 2

---

## 3. Dataset

Source: `marcosvolpato/edible-and-poisonous-fungi` (Kaggle)

The raw dataset ships with four folders which are consolidated into two clean classes:

| Class | Images | % of Dataset |
|-------|--------|-------------|
| Edible | 1,181 | 35% |
| Poisonous | 2,220 | 65% |

The 65/35 class imbalance is addressed with inverse-frequency weighted CrossEntropyLoss during training.

---

## 4. Notebook Cell Walkthrough

### Cell 1 — Install Dependencies
Installs all pip packages needed across the full pipeline. Run this first every session.


---

### Cell 2 — Import Libraries
Imports every library used across all subsequent cells: kagglehub, torch, torchvision, transformers, sklearn, PIL, and more.

---

### Cell 3 — Download Dataset
Downloads the Kaggle dataset via kagglehub and walks the downloaded directory to print the folder structure so you can verify the raw layout before consolidation.

---

### Cell 4 — Consolidate Folders
Clears any prior run artifacts under `/content/mushrooms/`, then merges the four raw source folders into two clean class folders:
- `edible sporocarp` + `edible mushroom sporocarp` -> `edible/`
- `poisonous sporocarp` + `poisonous mushroom sporocarp` -> `poisonous/`

Prints a file count per class to confirm the copy succeeded.

---

### Cell 5 — Transforms, Dataset Split & DataLoaders
Defines both transform pipelines, builds the dataset, and creates the DataLoaders.

Training augmentation:
- `Resize(256)` → `RandomCrop(224)`
- `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(45°)`
- `ColorJitter` (brightness, contrast, saturation, hue)
- ImageNet normalisation

Validation/inference transform: deterministic `Resize(224)` + ImageNet normalisation only.

Dataset split: 70% train / 15% validation / 15% test via `torch.utils.data.random_split`. DataLoaders use `batch_size=32`, `num_workers=2`.

---

Cell 6 — Load Model
Detects GPU/CPU, loads `microsoft/resnet-50` from HuggingFace with `num_labels=2`, and builds the `id2label` map (`0: edible`, `1: poisonous`).

---

### Cell 7 — Freeze Layers, Optimizer & Weighted Loss
- Freezes all layers except the final classifier head (~2,050 trainable params).
- AdamW optimizer with `lr=1e-4` on trainable parameters only.
- CrossEntropyLoss with inverse-frequency class weights to counter the 1,181 vs 2,220 imbalance.

---

### Cell 8 — Training Loop (30 Epochs)
Runs the main training loop. At each epoch:
1. Forward pass, loss computation, backprop, optimizer step (train phase).
2. Evaluate on validation set with `no_grad` (eval phase).
3. If validation loss improves, save weights to `/content/best_mushroom_resnet.pt`.

Training and validation loss/accuracy are printed every epoch. After the loop the best checkpoint is reloaded automatically. best_mushroom_resnet.pt is provided in the folder.

---

### Cell 9 — Load Weights from Local PC *(Optional)*
Use this cell when resuming in a new session after downloading `best_mushroom_resnet.pt`. Opens a Colab file-picker, uploads your `.pt` file, and loads it into the model so Cells 10 and 11 work without retraining.

---

### Cell 10 — Evaluation (Classification Report)
Runs the best-weight model over the full validation set and prints a `sklearn` `classification_report` with per-class precision, recall, and F1.

---

### Cell 11 — Single-Image Inference
Prompts you to upload any mushroom image from your computer (4 images provided: poisonous_1, poisonous_2, edible_1, edible_2. The cell will:
1. Display the image at 224×224.
2. Apply the validation transform and run a forward pass.
3. Compute softmax probabilities and display the predicted class: EDIBLE in green or POISONOUS in red with a confidence percentage.

---

## 5. Results

Validation Set Performance — Best Checkpoint (510 images)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 73% |
| Edible Precision | 63% |
| Edible Recall | 74% |
| Edible F1-Score | 0.68 |
| Poisonous Precision | 82% |
| Poisonous Recall | 73% |
| Poisonous F1-Score | 0.77 |
| Macro Avg F1 | 0.73 |
| Weighted Avg F1 | 0.74 |


---

## 6. Key Findings & Takeaways

### Strengths
- High poisonous precision (82%): When the model flags a mushroom as poisonous, it is correct 82% of the time critical for safety-oriented use.
- Solid poisonous recall (73%): The model catches nearly three-quarters of all poisonous mushrooms in the validation set.
- Fast fine-tuning: Freezing all but the classifier head kept trainable params to ~2,050, enabling 30 epochs on a free T4 GPU in under 10 minutes.
- Class-weighted loss: Prevented the model from collapsing to always predicting "poisonous" despite the 65/35 imbalance.

### Challenges
- Low edible precision (63%): The model over-predicts poisonous meaning many edible mushrooms are misclassified as poisonous (false positives). This is the dominant error type.
- Dataset size: ~3,400 images total is small for image classification. More data, especially diverse edible examples, would likely close the precision gap.
- Intra-class visual similarity: Edible and poisonous fungi can look nearly identical under different lighting, angles, and maturity stages, making visual-only classification very difficult.
- Single-stage fine-tuning: Only the classifier head was trained. Unfreezing deeper layers with a lower learning rate may lead to further gains.

---

## 7. Safety Disclaimer

This model is a proof of concept prototype only. Do NOT use it to make real-world foraging decisions.
A 73% accuracy rate means roughly 1 in 4 images is misclassified. Misidentifying a poisonous mushroom as edible can be fatal.
---

## 8. Quick-Start Guide
1. Open the notebook in Google Colab with GPU runtime enabled.
2. Run Cell 1 (install) -> Cell 2 (import).
3. Run Cell 3 (download dataset) -> Cell 4 (consolidate folders).
4. Run Cell 5 (transforms + split) -> Cell 6 (load model) → Cell 7 (freeze + optimizer).
5. Run Cell 8 (train ~10 min). Or download `best_mushroom_resnet.pt` from the folder and place in Cell 9. 
6. Run Cell 10 (evaluate) and Cell 11 (inference).

