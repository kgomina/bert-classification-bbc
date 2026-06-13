"""
model.py
--------
Chargement du modèle BERT pré-entraîné et définition de la tête de classification.
On utilise BertForSequenceClassification de Hugging Face qui ajoute automatiquement
une couche linéaire (dropout + linear) au-dessus du token [CLS].
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from pathlib import Path


# ---------------------------------------------------------------------------
# Nom du modèle pré-entraîné (anglais)
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_NAME = "bert-base-uncased"


def build_model(num_labels: int, dropout_prob: float = 0.1) -> BertForSequenceClassification:
    """
    Charge bert-base-uncased et ajoute une tête de classification linéaire.

    Architecture interne de BertForSequenceClassification :
        BERT encoder → pooler ([CLS] token) → Dropout → Linear(768, num_labels)

    Le dropout par défaut de BERT est déjà dans la config (hidden_dropout_prob=0.1),
    mais BertForSequenceClassification ajoute un dropout supplémentaire avant la tête.

    Args:
        num_labels   : nombre de classes (5 pour BBC News)
        dropout_prob : probabilité de dropout sur la tête de classification

    Returns:
        model : BertForSequenceClassification prêt pour le fine-tuning
    """
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_prob,
        attention_probs_dropout_prob=dropout_prob,
        # Supprime l'avertissement sur les poids non utilisés de la tête MLM
        ignore_mismatched_sizes=False,
    )
    print(f"[model] Modèle '{PRETRAINED_MODEL_NAME}' chargé ({num_labels} classes)")

    # Affichage du nombre de paramètres
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Paramètres totaux    : {total_params:,}")
    print(f"[model] Paramètres entraînables : {trainable_params:,}")

    return model


def load_tokenizer() -> BertTokenizer:
    """
    Charge le tokenizer BERT correspondant au modèle pré-entraîné.

    Le tokenizer gère :
      - La tokenization WordPiece (ex. "running" → ["run", "##ning"])
      - L'ajout des tokens spéciaux [CLS] et [SEP]
      - Le mapping token → ID via le vocabulaire de 30 522 tokens

    Returns:
        tokenizer : BertTokenizer configuré
    """
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    print(f"[model] Tokenizer chargé (vocab size = {tokenizer.vocab_size})")
    return tokenizer


def save_checkpoint(model: nn.Module, path: str, metadata: dict | None = None) -> None:
    """
    Sauvegarde le state_dict du modèle ainsi que des métadonnées optionnelles.

    On sauvegarde uniquement les poids (state_dict), pas l'objet modèle entier,
    ce qui est la bonne pratique PyTorch pour la portabilité.

    Args:
        model    : modèle PyTorch à sauvegarder
        path     : chemin de sortie (.pt ou .pth)
        metadata : dict optionnel (epoch, val_loss, val_acc, label_names…)
    """
    checkpoint = {"model_state_dict": model.state_dict()}
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)
    print(f"[model] Checkpoint sauvegardé → {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> dict:
    """
    Charge un checkpoint sauvegardé dans le modèle.

    Args:
        model  : instance du modèle (même architecture que lors de la sauvegarde)
        path   : chemin du fichier .pt
        device : device cible (cpu ou cuda)

    Returns:
        Le dict de métadonnées stocké dans le checkpoint (epoch, métriques…)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[model] Checkpoint chargé depuis {path}")
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
