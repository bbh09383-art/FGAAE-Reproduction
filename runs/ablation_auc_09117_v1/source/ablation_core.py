"""Small, auditable transformations of the frozen historical training source."""
from __future__ import annotations

import difflib
from pathlib import Path
import sys
import types

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "frozen/source"
ARMS = {
    "Base": {"adversarial": False, "features": False, "aqt": False},
    "A": {"adversarial": False, "features": True, "aqt": True},
    "B": {"adversarial": True, "features": False, "aqt": True},
    "C": {"adversarial": True, "features": True, "aqt": False},
    "Full": {"adversarial": True, "features": True, "aqt": True},
}


def transformed_source():
    original = (SOURCE / "fgaae.py").read_text(encoding="utf-8")
    text = original
    replacements = [
        ('critic_values.extend(model.critic(reconstructed).cpu().tolist())',
         'critic_values.extend(model.critic(reconstructed).cpu().tolist() if config.get("use_adversarial", True) else [0.0] * len(batch))', 1),
        ('(model.student(latent) - model.teacher(latent)).square().mean(dim=1).cpu().tolist()',
         '(model.student(latent) - model.teacher(latent)).square().mean(dim=1).cpu().tolist() if config.get("use_feature_guidance", True) else [0.0] * len(batch)', 1),
        ('model.critic.requires_grad_(True)', 'model.critic.requires_grad_(config.get("use_adversarial", True))', 2),
        ('for _ in range(config["critic_steps"]):', 'for _ in range(config["critic_steps"] if config.get("use_adversarial", True) else 0):', 1),
        ('adversarial_loss = -model.critic(generated).mean()',
         'adversarial_loss = -model.critic(generated).mean() if config.get("use_adversarial", True) else generated.new_zeros(())', 1),
        ('with torch.no_grad():\n                    teacher_features = model.teacher(latent)\n                distillation_loss = (model.student(latent) - teacher_features).square().mean()',
         'if config.get("use_feature_guidance", True):\n                    with torch.no_grad():\n                        teacher_features = model.teacher(latent)\n                    distillation_loss = (model.student(latent) - teacher_features).square().mean()\n                else:\n                    distillation_loss = generated.new_zeros(())', 1),
        ('decision_threshold = select_threshold(validation_result["scores"], validation_labels, config)\n            validation_metrics = metrics(validation_labels, validation_result["scores"], decision_threshold)',
         'validation_metrics = {"roc_auc": float(roc_auc_score(validation_labels, validation_result["scores"]))}', 1),
        ('                "validation_threshold": decision_threshold,\n', '', 1),
        ('for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]', 'for name in ["roc_auc"]', 1),
    ]
    for before, after, count in replacements:
        if text.count(before) != count:
            raise ValueError(f"Frozen-source transformation count changed: {before[:80]}")
        text = text.replace(before, after)
    patch = "".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile="frozen/fgaae.py", tofile="fgaae_ablation.py"))
    return text, patch


def load_module(name, text, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(text, str(filename), "exec"), module.__dict__)
    return module


def load_context():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    text, patch = transformed_source()
    core = load_module("fgaae_ablation", text, HERE / "fgaae_ablation.py")
    sys.modules["fgaae"] = core
    full = load_module("historical_full_model", (SOURCE / "full_model.py").read_text(encoding="utf-8"), SOURCE / "full_model.py")
    return core, full, text, patch


def configure(base, arm, seed, device, output):
    flags = ARMS[arm]
    config = {**base, "seed": seed, "device": str(device), "output": str(output),
              "use_aqt": flags["aqt"], "use_adversarial": flags["adversarial"],
              "use_feature_guidance": flags["features"], "early_stopping_patience": 0}
    if not flags["adversarial"]:
        config.update(adversarial_weight=0.0, critic_score_weight=0.0)
    if not flags["features"]:
        config["distillation_weight"] = 0.0
    # The frozen gradient monitor expects all three active loss graphs.
    config["monitor_encoder_gradients"] = bool(flags["adversarial"] and flags["features"])
    return config


def calibrate_encoder_bn(core, model, reference, batch_size, device):
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    model.eval()
    model.encoder.train()
    modules = []
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            modules.append((module, module.momentum))
            module.reset_running_stats()
            module.momentum = None
        elif isinstance(module, torch.nn.Dropout):
            module.eval()
    batches = 0
    with torch.no_grad():
        for (batch,) in core.make_loader(reference, batch_size):
            model.encoder(batch.to(device))
            batches += 1
    for module, momentum in modules:
        module.momentum = momentum
    model.eval()
    if not all(torch.equal(before[name], parameter.detach().cpu()) for name, parameter in model.named_parameters()):
        raise AssertionError("Unlabelled BN calibration changed parameters.")
    return {"method": "one deterministic pass; cumulative BN batch statistics; encoder Dropout disabled",
            "reference_rows": len(reference), "batches": batches, "batch_size": batch_size,
            "parameter_updates": 0, "expert_features_or_labels_used": False}


def auc_scores(core, model, values, config, device):
    result = core.evaluate(model, values, config, device)
    if not config.get("use_adversarial", True):
        if not np.array_equal(result["scores"], result["reconstruction_score"]):
            raise AssertionError("Removed Critic still affects scores.")
    return result
