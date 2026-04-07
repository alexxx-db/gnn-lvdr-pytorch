"""
Model evaluation utilities.

Training loop, evaluation loop, and metric computation for GraphSAGE.
Refactored from legacy Trainer and Evaluator classes into functional style
while preserving all computation logic.
"""
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, ones, zeros
from torch.nn import Sigmoid
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import mlflow

from src.graph.graphsage import Model
from src.config.settings import ProjectConfig
from src.mlops.tracking import log_training_step, log_evaluation_metrics


def compute_auc_ap(pos_score: torch.Tensor, neg_score: torch.Tensor) -> dict:
    """Compute AUC and Average Precision from positive/negative edge scores."""
    sigmoid = Sigmoid()
    pos_score_edge = sigmoid(pos_score)
    neg_score_edge = sigmoid(neg_score)

    scores = cat([pos_score_edge, neg_score_edge]).detach().numpy()
    labels = cat([
        ones(pos_score_edge.shape[0]),
        zeros(neg_score_edge.shape[0]),
    ]).detach().numpy()

    return {
        "AUC": roc_auc_score(labels.astype("int"), scores.squeeze()),
        "AP": average_precision_score(labels, scores),
    }


def compute_loss(pos_score, neg_score, loss_type: str = "binary_cross_entropy"):
    """Compute training loss from positive/negative scores."""
    n = pos_score.shape[0]

    if loss_type == "margin":
        sigmoid = Sigmoid()
        pos_prob = sigmoid(pos_score)
        neg_prob = sigmoid(neg_score)
        return (
            (neg_prob.view(n, -1) - pos_prob.view(n, -1) + 1)
            .clamp(min=0)
            .mean()
        )
    else:  # binary_cross_entropy
        # Use raw logits — bce_with_logits applies sigmoid internally
        scores = cat([pos_score, neg_score])
        labels = cat([ones(pos_score.shape[0]), zeros(neg_score.shape[0])])
        scores = scores.view(len(scores), -1).mean(dim=1)
        return F.binary_cross_entropy_with_logits(scores, labels)


def make_optimizer(model: Model, cfg: ProjectConfig):
    """Create optimizer from config."""
    lr = float(cfg.lr)
    wd = float(cfg.l2_regularisation)
    if cfg.optimiser == "SGD":
        return optim.SGD(model.parameters(), lr=lr,
                         momentum=cfg.momentum, weight_decay=wd)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def train_model(
    model: Model,
    train_loader,
    cfg: ProjectConfig,
) -> Model:
    """
    Train the GraphSAGE model.

    Preserves the legacy training loop: iterate through edge batches,
    compute loss, backprop, log metrics to MLflow.
    """
    opt = make_optimizer(model, cfg)
    model.train()

    with tqdm.tqdm(train_loader, desc="Training") as tq:
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
            if cfg.device == "cuda":
                blocks = [b.to(torch.device("cuda")) for b in blocks]
                positive_graph = positive_graph.to(torch.device("cuda"))
                negative_graph = negative_graph.to(torch.device("cuda"))

            input_features = blocks[0].srcdata["feature"]
            pos_score, neg_score = model(
                positive_graph=positive_graph,
                negative_graph=negative_graph,
                blocks=blocks,
                x=input_features,
            )
            loss = compute_loss(pos_score, neg_score, cfg.loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            results = compute_auc_ap(pos_score, neg_score)
            log_training_step(step, float(loss.item()), results["AUC"])
            tq.set_postfix({"loss": f"{loss.item():.3f}"}, refresh=False)

            if step >= cfg.num_epochs:
                break

    return model


def evaluate_model(
    model: Model,
    data_loader,
    split_name: str = "test",
) -> tuple[float, float]:
    """
    Evaluate the model on a data split.

    Returns (mean_auc, mean_ap) and logs to MLflow.
    """
    model.eval()
    auc_list, ap_list = [], []

    for input_nodes, positive_graph, negative_graph, blocks in data_loader:
        with torch.no_grad():
            input_features = blocks[0].srcdata["feature"]
            pos_score, neg_score = model(
                positive_graph=positive_graph,
                negative_graph=negative_graph,
                blocks=blocks,
                x=input_features,
            )
            metrics = compute_auc_ap(pos_score, neg_score)
            auc_list.append(metrics["AUC"])
            ap_list.append(metrics["AP"])

    mean_auc = float(np.mean(auc_list)) if auc_list else 0.0
    mean_ap = float(np.mean(ap_list)) if ap_list else 0.0

    log_evaluation_metrics(split_name, mean_auc, mean_ap)
    return mean_auc, mean_ap
