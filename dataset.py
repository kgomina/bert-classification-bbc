"""
dataset.py
----------
Classe Dataset PyTorch personnalisée pour la classification de texte.
Gère la tokenization BERT, le padding et les attention masks.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizer


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de textes avec BERT.

    Chaque item retourné est un dict contenant :
      - input_ids      : tenseur des token IDs (shape: [max_length])
      - attention_mask : masque binaire (1 = vrai token, 0 = padding) (shape: [max_length])
      - token_type_ids : segment IDs (tous à 0 pour tâche à phrase unique) (shape: [max_length])
      - label          : entier scalaire (indice de classe)
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        """
        Args:
            texts      : liste de textes bruts
            labels     : liste d'entiers représentant les classes
            tokenizer  : tokenizer Hugging Face (déjà chargé)
            max_length : longueur maximale de séquence en tokens
        """
        assert len(texts) == len(labels), "texts et labels doivent avoir la même longueur"
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Retourne le nombre total d'exemples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenize et encode un exemple à l'index idx.

        La méthode `encode_plus` du tokenizer gère automatiquement :
          - l'ajout des tokens spéciaux [CLS] et [SEP]
          - le padding jusqu'à max_length
          - la troncature si le texte dépasse max_length
          - la construction du masque d'attention

        Returns:
            dict avec 'input_ids', 'attention_mask', 'token_type_ids', 'label'
        """
        text  = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenization avec padding et troncature
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,   # ajoute [CLS] et [SEP]
            max_length=self.max_length,
            padding="max_length",      # padde jusqu'à max_length
            truncation=True,           # tronque si trop long
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",       # retourne des tenseurs PyTorch
        )

        return {
            # squeeze(0) : retire la dimension batch ajoutée par return_tensors="pt"
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Fonction utilitaire de chargement du dataset BBC News
# ---------------------------------------------------------------------------

def load_bbc_dataset(csv_path: str) -> tuple[list[str], list[int], list[str]]:
    """
    Charge le dataset BBC News depuis un fichier CSV (séparateur tabulation).

    Le dataset contient 4 colonnes : category, filename, title, content.
    On utilise la colonne 'content' comme texte et 'category' comme label.

    Args:
        csv_path : chemin vers le fichier bbc-news-data.csv

    Returns:
        texts       : liste de textes (content)
        labels      : liste d'entiers (indices de classes)
        label_names : liste des noms de classes (ordre alphabétique)
    """
    df = pd.read_csv(csv_path, sep="\t")

    # Suppression des éventuelles lignes avec valeurs manquantes
    df = df.dropna(subset=["content", "category"])
    df["content"] = df["content"].astype(str).str.strip()

    # Encodage des labels : tri alphabétique → reproductible
    label_names = sorted(df["category"].unique().tolist())
    label2id    = {name: idx for idx, name in enumerate(label_names)}

    texts  = df["content"].tolist()
    labels = [label2id[cat] for cat in df["category"].tolist()]

    print(f"[dataset] {len(texts)} exemples chargés, {len(label_names)} classes : {label_names}")
    return texts, labels, label_names
