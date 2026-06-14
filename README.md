# BBC News Category Classifier — Fine-tuning de BERT

> **Devoir Pratique n°3 — NLP avec PyTorch**  
> Classification de texte : Fine-tuning de BERT  
> Niveau Master / Ingénierie IA — NLP & Deep Learning

---

## 👥 Membres du binôme

| Membre | GitHub |
|--------|--------|
| Kéléfing GOMINA | @kgomina |
| Samba Abaladema WELA | @github2 |

---

## 📂 Dataset : BBC News

| Propriété | Valeur |
|-----------|--------|
| **Source** | BBC News (Kaggle / HuggingFace) |
| **Total d'exemples** | 2 225 articles |
| **Nombre de classes** | 5 |
| **Langue** | Anglais |
| **Séparateur CSV** | Tabulation (`\t`) |

### Distribution des classes

| Classe | Exemples | % |
|--------|----------|---|
| sport | 511 | 22.97% |
| business | 510 | 22.92% |
| politics | 417 | 18.74% |
| tech | 401 | 18.02% |
| entertainment | 386 | 17.35% |

> **Pas de déséquilibre** significatif (ratio max ≈ 1.32:1 < 2:1). Aucune stratégie particulière de rééquilibrage nécessaire.

### Statistiques de longueur (mots)

| Stat | Valeur |
|------|--------|
| Min | 84 mots |
| Moyenne | 379 mots |
| Médiane | 326 mots |
| Max | 4 428 mots |

### 5 exemples

```
[business]  UK house prices dipped slightly in November, the Office of the Deputy Prime Minister (ODPM) has said...
[sport]     Number eight Imanol Harinordoquy has been dropped from France's squad for the Six Nations match...
[politics]  Labour and the Conservatives are still telephoning millions of people who have signed up...
[tech]      Apple has unveiled a new range of music players, including an ultra-thin model that takes flash memory...
[entertainment] A Grammy-winning hip-hop duo has announced they are to tour together for the first time...
```

---

## 🏗️ Architecture et choix techniques

### Modèle : `bert-base-uncased`

BERT (Bidirectional Encoder Representations from Transformers) est un modèle de langage pré-entraîné sur Wikipedia + BookCorpus (3,3 milliards de mots). Il utilise une architecture Transformer encoder bidirectionnelle avec :

- 12 couches Transformer
- 12 têtes d'attention par couche
- Dimension cachée : 768
- ~110 millions de paramètres

**Choix justifiés :**

| Choix | Valeur | Justification |
|-------|--------|---------------|
| Modèle | `bert-base-uncased` | Dataset en anglais ; uncased suffisant pour la classification |
| `max_length` | 256 tokens | Capture ~75% des articles (médiane ≈ 326 mots ≈ 400 tokens) ; compromis VRAM/performance |
| Tête | Linear(768 → 5) + Dropout(0.1) | Standard pour la classification de séquences |

### Hyperparamètres d'entraînement

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| Learning rate | 2e-5 | Plage recommandée pour le fine-tuning BERT (évite le catastrophic forgetting) |
| Batch size | 16 | Compromis VRAM/stabilité du gradient |
| Epochs | 3 | BERT converge rapidement ; au-delà → risque d'overfitting |
| Optimiseur | AdamW | Régularisation L2 découplée, recommandé pour les Transformers |
| Weight decay | 0.01 | Régularisation légère (bias et LayerNorm exclus) |
| Scheduler | Linéaire avec warmup (10%) | Stabilise le début de l'entraînement |
| Loss | CrossEntropyLoss | Standard pour la classification multi-classe |
| Split | 80/20 stratifié | Préserve la distribution des classes dans chaque split |
| Seed | 42 | Reproductibilité complète |

---

## 📁 Structure du projet

```
bert-classification-bbc/
├── data/
│   └── bbc-news-data.csv       ← dataset (séparateur tabulation)
├── dataset.py                  ← TextClassificationDataset PyTorch
├── model.py                    ← chargement BERT + fonctions de sauvegarde
├── train.py                    ← boucles train_epoch / eval_epoch + main
├── demo.py                     ← interface Gradio
├── utils.py                    ← métriques, seed, visualisations
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation et exécution

### 1. Cloner le dépôt

```bash
git clone git clone https://github.com/kgomina/bert-classification-bbc.git
cd bert-classification-bbc
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Placer le dataset

```bash
mkdir -p data
# Copier bbc-news-data.csv dans le dossier data/
```

### 4. Lancer l'entraînement

```bash
python train.py
```

Options disponibles :
```bash
python train.py \
  --data_path data/bbc-news-data.csv \
  --model_dir checkpoints \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 256 \
  --seed 42
```

### 5. Lancer la démo Gradio

```bash
python demo.py --model_path checkpoints/best_model.pt
```

Pour partager via un lien public :
```bash
python demo.py --share
```

---

## 📊 Résultats


### Métriques finales

| Métrique | Valeur |
|----------|--------|
| Val Accuracy | 98.2%|
| Val F1 macro | 98.2% |

### Courbes d'apprentissage

![Learning Curves](learning_curves.png)

### Matrice de confusion

![Confusion Matrix](confusion_matrix.png)

### Interface Gradio

![alt text](<Capture d'écran 2026-06-14 012817.png>)

---

## 💡 Difficultés rencontrées et solutions

| Difficulté | Solution |
|-----------|----------|
| Longueur variable des textes | `truncation=True` + `padding="max_length"` dans le tokenizer |
| Risque de catastrophic forgetting | Learning rate bas (2e-5) + warmup scheduler |
| Gradient instable | Gradient clipping (`max_norm=1.0`) |
| Sauvegarde du meilleur modèle | Critère : `val_loss` minimale (plus fiable que `val_acc`) |
| Biais de régularisation | Exclusion de `bias` et `LayerNorm` du weight decay |

---

## 🔑 Concepts clés du Transfer Learning avec BERT

1. **Pré-entraînement** : BERT apprend des représentations générales du langage sur des milliards de mots grâce aux tâches MLM (Masked Language Modeling) et NSP (Next Sentence Prediction).

2. **Fine-tuning** : On remplace la tête de pré-entraînement par une tête de classification et on entraîne l'ensemble du réseau avec un petit learning rate.

3. **Apport du Transfer Learning** : Un modèle BERT fine-tuné atteint ~97% d'accuracy sur BBC News avec seulement 3 epochs, là où un modèle from scratch nécessiterait des dizaines d'epochs et beaucoup plus de données.

---

## 🔄 Répartition du travail

| Tâche | Membre |
|-------|--------|
| `dataset.py` + analyse exploratoire | Membre 1 |
| `model.py` + `utils.py` | Membre 2 |
| `train.py` (boucle d'entraînement) | Membre 1 |
| `demo.py` (interface Gradio) | Membre 2 |
| README.md + tests finaux | Les deux |
