# ISL CNN + KAN Research Project

This project implements a research-style Indian Sign Language image classification pipeline using a pretrained ResNet18 backbone and compares a standard linear classifier against a KAN classifier head.

## Research objective

Compare:
- **Baseline**: pretrained ResNet18 + linear classifier
- **Proposed**: pretrained ResNet18 + KAN head
- **Ablation**: pretrained ResNet18 + KAN head with last CNN block fine-tuning


## Dataset structure

Arrange your Kaggle ISL dataset like this:

```text
data/
  isl/
    train/
      A/
      B/
      ...
    val/
      A/
      B/
      ...
    test/
      A/
      B/
      ...
```

Each class folder should contain images for that class. The code uses `torchvision.datasets.ImageFolder`.

## Installation

```bash
pip install -r requirements.txt
```

## Run experiments

Baseline:

```bash
python train.py --config experiments/exp1_resnet18_baseline.yaml
```

KAN:

```bash
python train.py --config experiments/exp2_resnet18_kan.yaml
```

KAN + fine-tuning:

```bash
python train.py --config experiments/exp3_resnet18_kan_finetune.yaml
```

Evaluation:

```bash
python eval.py --config experiments/exp2_resnet18_kan.yaml --checkpoint results/exp2_resnet18_kan/best_model.pt
```

## Outputs

For each experiment, the project saves:
- `history.csv`
- `best_model.pt`
- `best_val_metrics.json`
- `test_metrics.json`
- validation and test confusion matrices

## Suggested paper-style analysis

Report the following:
- Accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision and recall
- Confusion matrix comparison between baseline and KAN
- Training stability and convergence behavior


