"""
Utility to evaluate any checkpoint on the validation split (865 images).
"""

import os
import sys
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import get_default_checkpoint_path
from ml.dataset_loader import EyeDiseaseDataset, CLASS_NAMES, NUM_CLASSES
from ml.eye_model import build_densenet201_model, load_eye_model_checkpoint, get_default_device
from ml.train_eye import evaluate_dataset

def evaluate_checkpoint_on_val(checkpoint_path: str = None):
    device = get_default_device()
    resolved_ckpt = checkpoint_path or get_default_checkpoint_path()
    print(f"Device: {device}")
    print(f"Evaluating: {resolved_ckpt}")
    
    manifest_path = os.path.join(os.path.dirname(__file__), "dataset", "split_manifest.json")
    val_ds = EyeDiseaseDataset(manifest_path, split="val")
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model = load_eye_model_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    val_eval = evaluate_dataset(model, val_loader, criterion, device, CLASS_NAMES)

    acc = val_eval["accuracy"]
    loss = val_eval["loss"]
    total = val_eval["total_samples"]
    correct = int(round(acc * total))
    
    from ml.experiment_finetune import compute_macro_f1_from_cm
    macro_f1, per_class = compute_macro_f1_from_cm(val_eval["confusion_matrix"])
    
    print("=" * 70)
    print(f"Validation Accuracy:    {acc * 100:.2f}% ({correct}/{total})")
    print(f"Validation Loss:        {loss:.4f}")
    print(f"Validation Macro F1:    {macro_f1 * 100:.2f}%")
    print("-" * 70)
    for c in CLASS_NAMES:
        m = per_class[c]
        print(f"  {c:<22}: Prec={m['precision']*100:5.2f}% | Rec={m['recall']*100:5.2f}% | F1={m['f1_score']*100:5.2f}% (Support: {m['support']})")
    print("=" * 70)
    print("Confusion Matrix:")
    for row in val_eval["confusion_matrix"]:
        print(f"  {row}")
    return {"accuracy": acc, "loss": loss, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": val_eval["confusion_matrix"]}

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_checkpoint_on_val(ckpt)
