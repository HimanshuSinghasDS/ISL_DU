from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    experiment_name: str = "resnet18_kan"
    data_dir: str = "data/isl"
    output_dir: str = "results"
    model_name: str = "resnet18_kan"  # resnet18_baseline | resnet18_kan
    num_classes: int = 26
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    seed: int = 42
    pretrained: bool = True
    freeze_backbone: bool = True
    fine_tune_last_block: bool = False
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    scheduler: str = "cosine"  # cosine | step | none
    step_size: int = 10
    gamma: float = 0.1
    early_stopping_patience: int = 7
    kan_hidden_dims: List[int] = field(default_factory=lambda: [256])
    dropout: float = 0.2
    use_class_weights: bool = False
    save_confusion_matrix: bool = True
    save_logits: bool = False
    device: Optional[str] = None
