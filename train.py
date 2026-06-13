"""
train.py
--------
Boucle d'entraînement PyTorch manuelle pour le fine-tuning de BERT.
Contient : train_epoch, eval_epoch, et la fonction main() complète.

Usage :
    python train.py --data_path data/bbc-news-data.csv --epochs 3
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import TextClassificationDataset, load_bbc_dataset
from model import build_model, load_tokenizer, save_checkpoint
from utils import set_seed, compute_metrics, plot_learning_curves, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut (modifiables via argparse)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    data_path   = "data/bbc-news-data.csv",
    model_dir   = "checkpoints",
    max_length  = 256,   # justification : médiane ~326 mots ≈ 400 tokens → 256 capture ~75% du texte
    batch_size  = 16,    # compromis VRAM / stabilité
    epochs      = 3,     # BERT converge vite ; au-delà → risque d'overfitting
    lr          = 2e-5,  # learning rate typique pour fine-tuning BERT
    weight_decay= 0.01,  # régularisation L2 via AdamW
    warmup_ratio= 0.1,   # 10% des steps en warmup linéaire
    seed        = 42,
    test_size   = 0.2,   # split 80/20 stratifié
)


# ---------------------------------------------------------------------------
# Boucle d'entraînement (une epoch)
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Exécute une epoch d'entraînement complète.

    Étapes pour chaque batch :
      1. Forward pass : calcul des logits
      2. Calcul de la loss (CrossEntropyLoss)
      3. Backward pass : calcul des gradients
      4. Gradient clipping (évite l'explosion des gradients)
      5. Mise à jour des poids (optimizer.step)
      6. Mise à jour du scheduler de learning rate
      7. Remise à zéro des gradients (optimizer.zero_grad)

    Args:
        model     : BertForSequenceClassification en mode train
        loader    : DataLoader d'entraînement
        optimizer : AdamW
        scheduler : scheduler linéaire avec warmup
        device    : cpu ou cuda
        criterion : CrossEntropyLoss

    Returns:
        avg_loss : loss moyenne sur l'epoch
        accuracy : accuracy sur l'epoch
    """
    model.train()  # active le dropout et BatchNorm

    total_loss  = 0.0
    correct     = 0
    total       = 0

    # tqdm affiche la barre de progression dans le terminal
    progress = tqdm(loader, desc="  [Train]", leave=False)

    for batch in progress:
        # Transfert des données vers le bon device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        # --- Forward pass ---
        # BertForSequenceClassification retourne un objet SequenceClassifierOutput
        # qui contient : loss (si labels fournis), logits, hidden_states, attentions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits  # shape : (batch_size, num_labels)

        # --- Calcul de la loss ---
        # CrossEntropyLoss attend des logits (non-softmaxés) et des labels entiers
        loss = criterion(logits, labels)

        # --- Backward pass ---
        loss.backward()

        # Gradient clipping : empêche les gradients trop grands (max_norm=1.0)
        # Particulièrement important pour BERT avec son grand nombre de couches
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Mise à jour ---
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # --- Statistiques ---
        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Affichage live dans la barre de progression
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Boucle d'évaluation (validation)
# ---------------------------------------------------------------------------

def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, list, list]:
    """
    Évalue le modèle sur le loader de validation.

    IMPORTANT : on bascule le modèle en mode eval (model.eval()) et on
    désactive le calcul des gradients (torch.no_grad()) pour :
      - Désactiver le dropout (reproductibilité des prédictions)
      - Ne pas construire le graphe de calcul → économie de mémoire

    Args:
        model     : BertForSequenceClassification
        loader    : DataLoader de validation
        device    : cpu ou cuda
        criterion : CrossEntropyLoss

    Returns:
        avg_loss : loss moyenne
        accuracy : accuracy
        f1_macro : F1-score macro
        all_preds  : liste de toutes les prédictions
        all_labels : liste de tous les vrais labels
    """
    model.eval()  # désactive dropout

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():  # pas de calcul de gradients pendant la validation
        progress = tqdm(loader, desc="  [Val]  ", leave=False)
        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1_macro"], all_preds, all_labels


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """
    Orchestre l'entraînement complet :
      1. Fixation de la seed
      2. Chargement du dataset et split stratifié
      3. Construction des DataLoaders
      4. Initialisation du modèle, optimiseur, scheduler, loss
      5. Boucles d'entraînement et de validation
      6. Sauvegarde du meilleur modèle (best val_loss)
      7. Rapport final et visualisations
    """
    # --- 1. Reproductibilité ---
    set_seed(args.seed)

    # --- 2. Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # --- 3. Chargement des données ---
    texts, labels, label_names = load_bbc_dataset(args.data_path)
    num_labels = len(label_names)

    # Split stratifié : conserve la distribution des classes dans chaque split
    # stratify=labels garantit que chaque classe est représentée proportionnellement
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    print(f"[main] Split → Train : {len(X_train)} | Val : {len(X_val)}")

    # --- 4. Tokenizer ---
    tokenizer = load_tokenizer()

    # --- 5. Datasets PyTorch ---
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset   = TextClassificationDataset(X_val,   y_val,   tokenizer, args.max_length)

    # DataLoaders : shuffle=True pour le train, False pour la val
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- 6. Modèle ---
    model = build_model(num_labels=num_labels)
    model = model.to(device)

    # --- 7. Optimiseur AdamW ---
    # On exclut le bias et les LayerNorm des la régularisation L2 (bonne pratique)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=args.lr)

    # --- 8. Scheduler linéaire avec warmup ---
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[main] Steps totaux : {total_steps} | Warmup : {warmup_steps}")

    # --- 9. Fonction de perte ---
    # CrossEntropyLoss attend des logits (pas de softmax) et des labels entiers
    criterion = nn.CrossEntropyLoss()

    # --- 10. Dossier de sauvegarde ---
    os.makedirs(args.model_dir, exist_ok=True)
    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    best_val_loss   = float("inf")

    # --- 11. Historique pour les courbes ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_f1": [],
    }

    # --- 12. Boucle d'entraînement ---
    print(f"\n{'='*60}")
    print(f"  Début de l'entraînement ({args.epochs} epochs)")
    print(f"{'='*60}\n")

    final_preds  = []
    final_labels = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )

        # Validation
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(
            model, val_loader, device, criterion
        )

        # Mise à jour de l'historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Learning rate courant (premier groupe de paramètres)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  train_loss={train_loss:.4f} | train_acc={train_acc:.4f}\n"
            f"  val_loss  ={val_loss:.4f}   | val_acc  ={val_acc:.4f}  | val_f1={val_f1:.4f}\n"
            f"  lr={current_lr:.2e}"
        )

        # Sauvegarde du meilleur modèle (critère : val_loss minimale)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_preds   = val_preds
            final_labels  = val_labels
            save_checkpoint(
                model,
                best_model_path,
                metadata={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "label_names": label_names,
                    "num_labels": num_labels,
                    "max_length": args.max_length,
                },
            )
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        print()

    # --- 13. Rapport final ---
    print(f"\n{'='*60}")
    print("  Rapport final (meilleur modèle)")
    print(f"{'='*60}")
    metrics = compute_metrics(final_labels, final_preds, label_names)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print("\n" + metrics["report"])

    # --- 14. Visualisations ---
    plot_learning_curves(history, save_path=os.path.join(args.model_dir, "learning_curves.png"))
    plot_confusion_matrix(
        final_labels, final_preds, label_names,
        save_path=os.path.join(args.model_dir, "confusion_matrix.png")
    )

    print(f"\n[main] Entraînement terminé. Modèle → {best_model_path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BERT pour BBC News")
    parser.add_argument("--data_path",    type=str,   default=DEFAULTS["data_path"])
    parser.add_argument("--model_dir",    type=str,   default=DEFAULTS["model_dir"])
    parser.add_argument("--max_length",   type=int,   default=DEFAULTS["max_length"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--test_size",    type=float, default=DEFAULTS["test_size"])

    args = parser.parse_args()

    # Affichage de la configuration
    print("\nConfiguration :")
    for k, v in vars(args).items():
        print(f"  {k:15s} = {v}")
    print()

    main(args)
