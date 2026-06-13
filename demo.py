"""
demo.py
-------
Interface de démonstration interactive avec Gradio.
Charge le meilleur modèle entraîné et prédit la catégorie d'un texte BBC News.

Usage :
    python demo.py
    python demo.py --model_path checkpoints/best_model.pt
"""

import argparse
import torch
import torch.nn.functional as F
import gradio as gr

from model import build_model, load_tokenizer, load_checkpoint, PRETRAINED_MODEL_NAME


# ---------------------------------------------------------------------------
# Chargement du modèle (fait une seule fois au démarrage de l'interface)
# ---------------------------------------------------------------------------

def load_model_for_demo(model_path: str, device: torch.device):
    """
    Charge le checkpoint sauvegardé et retourne le modèle, le tokenizer
    et les noms de classes.

    Args:
        model_path : chemin vers best_model.pt
        device     : cpu ou cuda

    Returns:
        model, tokenizer, label_names, max_length
    """
    # Chargement du checkpoint pour récupérer les métadonnées
    checkpoint = torch.load(model_path, map_location=device)
    label_names = checkpoint["label_names"]
    num_labels  = checkpoint["num_labels"]
    max_length  = checkpoint.get("max_length", 256)

    # Construction du modèle avec la bonne architecture
    model = build_model(num_labels=num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # mode évaluation : désactive le dropout

    tokenizer = load_tokenizer()

    print(f"[demo] Modèle chargé : {num_labels} classes → {label_names}")
    print(f"[demo] Max length    : {max_length}")

    return model, tokenizer, label_names, max_length


# ---------------------------------------------------------------------------
# Fonction de prédiction (appelée par Gradio)
# ---------------------------------------------------------------------------

def predict(
    text: str,
    model: torch.nn.Module,
    tokenizer,
    label_names: list[str],
    max_length: int,
    device: torch.device,
) -> dict:
    """
    Prédit la catégorie d'un texte et retourne les probabilités par classe.

    Args:
        text        : texte saisi par l'utilisateur
        model       : modèle fine-tuné en mode eval
        tokenizer   : tokenizer BERT
        label_names : noms des classes
        max_length  : longueur maximale de séquence
        device      : cpu ou cuda

    Returns:
        dict {label: probabilité} pour l'affichage Gradio
    """
    if not text.strip():
        return {label: 0.0 for label in label_names}

    # Tokenization du texte
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    # Inférence sans calcul de gradients
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits  # shape : (1, num_labels)

        # Softmax pour obtenir des probabilités (somme = 1)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Construction du dict Gradio : {label: float}
    return {label: float(probs[i]) for i, label in enumerate(label_names)}


# ---------------------------------------------------------------------------
# Exemples pré-remplis
# ---------------------------------------------------------------------------

EXAMPLES = [
    [
        "Arsenal secured a dramatic late victory at Stamford Bridge thanks to a stunning "
        "90th-minute goal from their striker, moving them to the top of the Premier League table."
    ],
    [
        "The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation "
        "concerns. Markets reacted nervously as investors reassessed their portfolios."
    ],
    [
        "Apple unveiled its latest iPhone model featuring an advanced AI chip and improved camera "
        "system. The new device is expected to dominate smartphone sales this holiday season."
    ],
    [
        "The Prime Minister announced a major reshuffle of the cabinet following the party's "
        "disappointing performance in the local elections held across England and Wales."
    ],
    [
        "The acclaimed director's new film has received five nominations at the upcoming awards "
        "ceremony. Critics praised the lead actress for her captivating performance."
    ],
]


# ---------------------------------------------------------------------------
# Construction et lancement de l'interface Gradio
# ---------------------------------------------------------------------------

def build_interface(model, tokenizer, label_names, max_length, device):
    """
    Construit l'interface Gradio avec :
      - Un champ texte pour la saisie
      - Un composant Label pour afficher les probabilités
      - Des exemples pré-remplis
    """

    # Fonction wrapper pour Gradio (ne prend qu'un argument)
    def gradio_predict(text: str) -> dict:
        return predict(text, model, tokenizer, label_names, max_length, device)

    with gr.Blocks(
        title="BBC News Classifier - BERT",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📰 BBC News Category Classifier
            ## Fine-tuned BERT (`bert-base-uncased`) sur BBC News Dataset
            
            Ce modèle classe automatiquement un article de presse en anglais dans l'une
            des **5 catégories BBC** : `business`, `entertainment`, `politics`, `sport`, `tech`.
            
            **Comment utiliser :** Collez ou saisissez un texte dans la zone ci-dessous,
            puis cliquez sur **Classifier**.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Texte à classifier",
                    placeholder="Entrez un article de presse en anglais...",
                    lines=8,
                )
                classify_btn = gr.Button("🔍 Classifier", variant="primary")

            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Catégories prédites (probabilités)",
                    num_top_classes=5,
                )

        # Lancement via bouton ou touche Entrée
        classify_btn.click(fn=gradio_predict, inputs=input_text, outputs=output_label)
        input_text.submit(fn=gradio_predict,  inputs=input_text, outputs=output_label)

        gr.Markdown("### 📌 Exemples")
        gr.Examples(
            examples=EXAMPLES,
            inputs=input_text,
            outputs=output_label,
            fn=gradio_predict,
            cache_examples=True,
        )

        gr.Markdown(
            """
            ---
            **Modèle :** `bert-base-uncased` fine-tuné sur 1780 articles BBC News  
            **Classes :** business · entertainment · politics · sport · tech  
            **Hyperparamètres :** lr=2e-5, batch=16, epochs=3, max_length=256
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démo Gradio - BBC News Classifier")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/best_model.pt",
        help="Chemin vers le fichier best_model.pt sauvegardé par train.py"
    )
    parser.add_argument("--share", action="store_true", help="Partager via lien public Gradio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device : {device}")

    model, tokenizer, label_names, max_length = load_model_for_demo(args.model_path, device)

    demo = build_interface(model, tokenizer, label_names, max_length, device)
    demo.launch(share=args.share)
