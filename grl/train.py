"""Training utilities and research training loops.

This module should contain the reusable parts of training that scripts call:
configuration handling, model construction, loss calculation, checkpointing, and
the training loop itself.

It does not need to become a general experiment framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from grl.data import LabeledPointSetDataset
from grl.models import CoordinateKnnGraphPointScorer, FlatSetMlpPointScorer


def beta_union_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Collapse possibly tied support labels into one point-level target."""
    beta = batch["beta"].float()
    support_count = batch["support_count"].long()
    target = torch.zeros_like(beta[:, 0, :])
    for batch_index, count in enumerate(support_count.tolist()):
        target[batch_index] = beta[batch_index, :count].amax(dim=0)
    return target


def alpha_union_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Collapse possibly tied support labels into one coordinate-level target."""
    alpha = batch["alpha"].float()
    support_count = batch["support_count"].long()
    target = torch.zeros_like(alpha[:, 0, :, :])
    for batch_index, count in enumerate(support_count.tolist()):
        target[batch_index] = alpha[batch_index, :count].amax(dim=0)
    return target


def model_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    *,
    alpha_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute beta point loss plus optional alpha coordinate loss."""
    points = batch["points"].to(device).float()
    beta_target = beta_union_target(batch).to(device)
    logits = model(points)
    beta_loss = loss_fn(logits, beta_target)
    loss = beta_loss
    metrics = {"beta_loss": float(beta_loss.detach().cpu())}

    coordinate_logits = getattr(model, "last_coordinate_logits", None)
    if alpha_weight and coordinate_logits is not None:
        alpha_target = alpha_union_target(batch).to(device)
        alpha_loss = loss_fn(coordinate_logits, alpha_target)
        loss = loss + alpha_weight * alpha_loss
        metrics["alpha_loss"] = float(alpha_loss.detach().cpu())
    else:
        metrics["alpha_loss"] = 0.0

    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    alpha_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str,
    show_progress: bool = True,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "beta_loss": 0.0, "alpha_loss": 0.0}
    total_items = 0
    iterator = tqdm(loader, desc=description, leave=False, disable=not show_progress)

    for batch in iterator:
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            loss, metrics = model_losses(
                model,
                batch,
                loss_fn,
                alpha_weight=alpha_weight,
                device=device,
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size_actual = int(batch["points"].shape[0])
        total_items += batch_size_actual
        for key in totals:
            totals[key] += metrics[key] * batch_size_actual
        iterator.set_postfix(loss=totals["loss"] / total_items)

    return {key: value / total_items for key, value in totals.items()}


def make_model(
    model_name: str,
    *,
    input_n: int | None = None,
    input_d: int | None = None,
    hidden_dim: int = 64,
    graph_layers: int = 0,
    graph_k: int = 8,
    graph_metric: str = "euclidean",
    graph_kind: str | None = None,
    coordinate_shared: bool = True,
) -> torch.nn.Module:
    """Construct a point scorer by name."""
    if model_name == "graph":
        return CoordinateKnnGraphPointScorer(
            hidden_dim=hidden_dim,
            num_layers=graph_layers,
            k=graph_k,
            metric=graph_metric,
            graph_kind=graph_kind,
            coordinate_shared=coordinate_shared,
            input_dim=input_d,
        )
    if model_name == "flat_mlp":
        if input_n is None or input_d is None:
            raise ValueError("input_n and input_d are required for flat_mlp")
        return FlatSetMlpPointScorer(n=input_n, d=input_d, hidden_dim=hidden_dim)
    raise ValueError(f"unknown model: {model_name}")


def train_point_scorer(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    val_dataset_path: str | Path | None = None,
    model_name: str = "graph",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    alpha_weight: float = 1.0,
    graph_layers: int = 0,
    graph_k: int = 8,
    graph_metric: str = "euclidean",
    graph_kind: str | None = None,
    coordinate_shared: bool = True,
    seed: int = 0,
    device: str | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train a point scorer and save a checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if graph_layers == 0:
        graph_k = 0
    resolved_graph_kind = graph_kind or f"knn_{graph_metric}"

    torch.manual_seed(seed)
    dataset = LabeledPointSetDataset(dataset_path)
    input_n = int(dataset.arrays["points"].shape[1])
    input_d = int(dataset.arrays["points"].shape[2])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(LabeledPointSetDataset(val_dataset_path), batch_size=batch_size)
        if val_dataset_path is not None
        else None
    )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = make_model(
        model_name,
        input_n=input_n,
        input_d=input_d,
        hidden_dim=hidden_dim,
        graph_layers=graph_layers,
        graph_k=graph_k,
        graph_metric=graph_metric,
        graph_kind=resolved_graph_kind,
        coordinate_shared=coordinate_shared,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_state_dict = None
    best_val_loss = float("inf")

    epoch_iterator = tqdm(
        range(1, epochs + 1),
        desc="epochs",
        disable=not show_progress,
    )
    for epoch in epoch_iterator:
        train_metrics = run_epoch(
            model,
            loader,
            loss_fn,
            alpha_weight=alpha_weight,
            device=resolved_device,
            optimizer=optimizer,
            description=f"train {epoch}/{epochs}",
            show_progress=show_progress,
        )
        record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                alpha_weight=alpha_weight,
                device=resolved_device,
                optimizer=None,
                description=f"val {epoch}/{epochs}",
                show_progress=show_progress,
            )
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        history.append(record)
        postfix = {"train_loss": record["train_loss"]}
        if "val_loss" in record:
            postfix["val_loss"] = record["val_loss"]
        epoch_iterator.set_postfix(postfix)

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    checkpoint = {
        "model": model_name,
        "input_n": input_n,
        "input_d": input_d,
        "hidden_dim": hidden_dim,
        "alpha_weight": alpha_weight,
        "graph_layers": graph_layers,
        "graph_k": graph_k,
        "graph_metric": graph_metric,
        "graph_kind": resolved_graph_kind,
        "coordinate_shared": coordinate_shared,
        "state_dict": model.state_dict(),
        "best_state_dict": best_state_dict,
        "best_metric": "val_loss" if val_dataset_path is not None else "train_loss",
        "best_value": best_val_loss if val_dataset_path is not None else history[-1]["train_loss"],
        "history": history,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint
