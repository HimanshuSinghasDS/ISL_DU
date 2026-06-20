import argparse
from pathlib import Path

import torch
import torch.nn as nn

from datasets import get_dataloaders
from train import build_model, load_config, run_epoch
from utils import save_confusion_matrix_plot, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, class_names = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.image_size)

    cfg.num_classes = len(class_names)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    metrics, y_true, y_pred = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)

    out_dir = Path(cfg.output_dir) / cfg.experiment_name / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / 'metrics.json', metrics)
    save_confusion_matrix_plot(y_true, y_pred, class_names, out_dir / 'confusion_matrix.png')


if __name__ == '__main__':
    main()
