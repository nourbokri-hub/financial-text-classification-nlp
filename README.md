# Classification de textes financiers – NLP

Projet réalisé lors de mon stage chez **NEOLedge**.

L'objectif est de classifier des phrases issues des communications de la Banque centrale européenne (BCE) en quatre catégories : **hawkish, dovish, neutral et irrelevant**.

## Modèles utilisés

J'ai travaillé avec trois modèles de Deep Learning :
- **RNN**
- **LSTM**
- **DistilBERT** avec Transfer Learning (modèle gelé) puis **fine-tuning complet**

J'ai comparé leurs performances à une baseline et testé différentes techniques pour limiter le surapprentissage, notamment le Dropout, la régularisation et l'Early Stopping.

## Dataset

Le dataset provient de [Hugging Face](https://huggingface.co/datasets/gtfintechlab/european_central_bank).

Il contient 1 000 phrases réparties en :
- 700 pour l'entraînement
- 150 pour la validation
- 150 pour le test

## Résultats

Résultats sur le jeu de test (150 phrases) :

| Modèle | Accuracy | F1-score pondéré | AUC |
|---|---:|---:|---:|
| Baseline (classe majoritaire) | 34,67 % | 17,85 % | – |
| RNN amélioré | 26,00 % | 14,20 % | 49,96 % |
| LSTM amélioré | 32,67 % | 24,51 % | 51,40 % |
| DistilBERT gelé | 51,33 % | 50,12 % | 70,92 % |
| **DistilBERT fine-tuné** | **54,00 %** | **52,70 %** | **73,97 %** |

Le DistilBERT fine-tuné obtient les meilleurs résultats, avec un F1-score pondéré environ 3 fois supérieur à celui de la baseline. Les modèles RNN et LSTM, entraînés à partir de zéro sur seulement 700 phrases, ne dépassent pas la baseline en accuracy : ce résultat montre l'intérêt du transfer learning sur un petit jeu de données.

J'ai également développé une petite interface avec **Gradio** pour tester les prédictions.

## Technologies

Python, PyTorch, Hugging Face Transformers, scikit-learn, Pandas, Matplotlib et Gradio.

## Installation

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

Ouvrir ensuite `financial_text_classification.ipynb` dans VS Code ou Jupyter Notebook et exécuter les cellules dans l'ordre.

Les données sont téléchargées directement depuis Hugging Face. L'entraînement des modèles peut prendre du temps : un GPU (par exemple sur Google Colab) est recommandé pour le fine-tuning de DistilBERT.