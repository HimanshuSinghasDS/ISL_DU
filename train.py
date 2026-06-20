import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from config import ExperimentConfig
from datasets import get_class_weights, get_dataloaders
from models import ResNet18Baseline, ResNet18KAN
from utils import compute_metrics, ensure_dir, save_confusion_matrix_plot, save_json, save_metrics_row, set_seed


def load_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    cfg = ExperimentConfig(**data)
    return cfg


def build_model(cfg):
    if cfg.model_name == 'resnet18_baseline':
        model = ResNet18Baseline(cfg.num_classes, cfg.pretrained, cfg.freeze_backbone, cfg.dropout)
    elif cfg.model_name == 'resnet18_kan':
        model = ResNet18KAN(cfg.num_classes, cfg.pretrained, cfg.freeze_backbone, cfg.kan_hidden_dims)
    else:
        raise ValueError(f'Unknown model_name: {cfg.model_name}')

    if cfg.fine_tune_last_block:
        model.unfreeze_last_block()
    return model


def get_optimizer(cfg, model):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    params = []
    if backbone_params:
        params.append({'params': backbone_params, 'lr': cfg.backbone_lr})
    if head_params:
        params.append({'params': head_params, 'lr': cfg.head_lr})
    return torch.optim.AdamW(params, weight_decay=cfg.weight_decay)


def get_scheduler(cfg, optimizer):
    if cfg.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    return None


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train and hasattr(model, 'regularization_loss'):
                reg = model.regularization_loss()
                if not isinstance(reg, float):
                    loss = loss + 1e-4 * reg
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(cfg.output_dir) / cfg.experiment_name
    ensure_dir(run_dir)
    save_json(run_dir / 'config.json', asdict(cfg))

    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.image_size)

    if cfg.num_classes != len(class_names):
        print(f'[Info] Overriding num_classes from {cfg.num_classes} to {len(class_names)} based on dataset folders.')
        cfg.num_classes = len(class_names)

    model = build_model(cfg).to(device)

    weights = None
    if cfg.use_class_weights:
        weights = get_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    best_f1 = -1
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics, val_true, val_pred = run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)

        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_macro_f1': train_metrics['macro_f1'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
        }
        save_metrics_row(run_dir / 'history.csv', row)

        if scheduler is not None:
            scheduler.step()

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            patience = 0
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'config': asdict(cfg)}, run_dir / 'best_model.pt')
            save_json(run_dir / 'best_val_metrics.json', val_metrics)
            if cfg.save_confusion_matrix:
                save_confusion_matrix_plot(val_true, val_pred, class_names, run_dir / 'val_confusion_matrix.png')
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print('Early stopping triggered.')
                break

    checkpoint = torch.load(run_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics, test_true, test_pred = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    save_json(run_dir / 'test_metrics.json', test_metrics)
    if cfg.save_confusion_matrix:
        save_confusion_matrix_plot(test_true, test_pred, class_names, run_dir / 'test_confusion_matrix.png')

    summary = {
        'experiment_name': cfg.experiment_name,
        'best_val_macro_f1': best_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_weighted_f1': test_metrics['weighted_f1']
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
