"""
utils.py
--------
Fonctions utilitaires : fixation de seed, calcul de métriques,
et visualisations des courbes d'apprentissage + matrice de confusion.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fixe la graine aléatoire pour random, numpy et torch (CPU + GPU).
    À appeler en tout premier dans train.py pour garantir la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Rend les opérations CUDA déterministes (peut ralentir l'entraînement)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """
    Calcule accuracy et F1-score macro à partir des labels réels et prédits.

    Args:
        y_true      : liste des labels réels (entiers)
        y_pred      : liste des labels prédits (entiers)
        label_names : noms des classes pour le rapport détaillé

    Returns:
        dict contenant 'accuracy', 'f1_macro', et le rapport textuel
    """
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    )
    return {"accuracy": acc, "f1_macro": f1, "report": report}


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_learning_curves(history: dict, save_path: str = "learning_curves.png") -> None:
    """
    Trace et sauvegarde les courbes loss et accuracy (train vs validation).

    Args:
        history   : dict avec clés 'train_loss', 'val_loss',
                    'train_accuracy', 'val_accuracy'
        save_path : chemin de sortie de l'image
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss par epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --- Accuracy ---
    axes[1].plot(epochs, history["train_accuracy"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_accuracy"],   "r-o", label="Val Accuracy")
    axes[1].set_title("Accuracy par epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Courbes d'apprentissage - BERT BBC News", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Courbes sauvegardées → {save_path}")


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    save_path: str = "confusion_matrix.png",
) -> None:
    """
    Trace et sauvegarde la matrice de confusion normalisée.

    Args:
        y_true      : labels réels
        y_pred      : labels prédits
        label_names : noms des classes
        save_path   : chemin de sortie
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalisation par ligne

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title("Matrice de confusion (normalisée)")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Matrice de confusion sauvegardée → {save_path}")
