# Classification de textes financiers – NLP

Projet réalisé lors de mon stage chez **NEOLedge**.

L'objectif est de classifier des phrases issues des communications de la Banque centrale européenne (BCE) en quatre catégories : **hawkish, dovish, neutral et irrelevant**.

## Modèles utilisés

J'ai travaillé avec trois modèles de Deep Learning :
- **RNN**
- **LSTM**
- **DistilBERT** avec Transfer Learning

J'ai comparé leurs performances et testé différentes techniques pour limiter le surapprentissage, notamment le Dropout, la régularisation et l'Early Stopping.

## Dataset

Le dataset provient de [Hugging Face](https://huggingface.co/datasets/gtfintechlab/european_central_bank).

Il contient 1 000 phrases réparties en :
- 700 pour l'entraînement
- 150 pour la validation
- 150 pour le test

## Résultats

| Modèle | Accuracy | F1-score |
|---|---:|---:|
| RNN amélioré | 26,67 % | 15,52 % |
| LSTM amélioré | 34,67 % | 24,68 % |
| DistilBERT | 52,67 % | 51,04 % |

DistilBERT a obtenu les meilleurs résultats parmi les modèles comparés.

J'ai également développé une petite interface avec **Gradio** pour tester les prédictions.

## Technologies

Python, PyTorch, Hugging Face Transformers, scikit-learn, Pandas, Matplotlib et Gradio.

## Installation

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

Ouvrir ensuite `financial_text_classification.ipynb` dans VS Code ou Jupyter Notebook et exécuter les cellules dans l'ordre.

Les données sont téléchargées directement depuis Hugging Face. L'entraînement des modèles peut prendre du temps.