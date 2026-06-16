#  Projet Deep Learning - MLP, CNN et RNN

##  Description du Projet
Ce projet explore trois architectures de Deep Learning sur différentes tâches :
- **Partie 1 (MLP)** : Classification binaire sur le dataset Breast Cancer avec étude des initialisations de poids
- **Partie 2 (CNN)** : Classification d'images Fashion-MNIST avec comparaison CNN vs MLP et variantes architecturales
- **Partie 3 (RNN)** : Génération de texte et traduction automatique Seq2Seq

##  Structure du Repository
DeepLearning/
├── README.md # Ce fichier
├── part1Mlp.ipynb # Notebook MLP
├── part2_cnn.ipynb # Notebook CNN
├── part3_rnn.ipynb # Notebook RNN/Seq2Seq
├── best_mlp_gaussian.pth # Meilleur modèle MLP (Gaussian)
├── best_mlp_constant.pth # Modèle MLP (Constant)
├── best_mlp_xavier.pth # Modèle MLP (Xavier)
├── best_mlp_part1.pth # Modèle MLP de base
├── architectural_variants_comparison.csv # Comparaison variantes CNN
├── data/ # Données Fashion-MNIST
├── part2_results/ # Résultats Partie 2
│ └── figures/
│ ├── mlp_vs_cnn_comparison.png
│ ├── architectural_variants.png
│ └── feature_maps_*.png
└── part3_rnn/ # Résultats Partie 3
├── figures/
│ └── loss_perplexity_comparison.png
└── results/
├── seq2seq_loss.png
└── translation_results.csv


## ⚙️ Prérequis
```bash
pip install torch torchvision scikit-learn matplotlib pandas numpy

 ---Partie 1 : MLP - Classification du Cancer du Sein
Dataset
Source : sklearn.datasets.load_breast_cancer
Échantillons : 569 (30 features chacune)
Classes : 0 (maligne) / 1 (bénigne)
Split : 70% train / 15% val / 15% test (stratifié)
Prétraitement : StandardScaler + conversion en tenseurs PyTorch
Architecture:
Input (30) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → BCEWithLogitsLoss

--Étude des Initialisations
Trois stratégies comparées sur 50 epochs (lr=1e-3, batch_size=32) :
Résultats Finaux (Meilleur modèle : Gaussian)

---Partie 2 : CNN - Classification Fashion-MNIST

Dataset
Source : torchvision.datasets.FashionMNIST
Images : 28×28 pixels, 1 canal, 10 classes
Normalisation : mean=0.2860, std=0.3530
Split : 85% train / 15% val / test séparé (batch_size=128)
Implémentation From Scratch
✅ Corrélation 2D manuelle validée vs nn.Conv2d
✅ Max-Pooling 2×2 manuel validé vs F.max_pool2d
✅ Average-Pooling 2×2 manuel validé vs F.avg_pool2d
✅ Détection de bords (filtre Sobel) appliquée sur une image réelle
Comparaison MLP vs CNN (5 epochs)
 Le CNN atteint la même performance avec 2.5× moins de paramètres !
 ---Partie 3 : RNN - Génération et Traduction

--3.1 Génération de Texte (Character-level)

Dataset : Extrait de Roméo et Juliette de Shakespeare (625 caractères, vocabulaire de 41 caractères)
Architecture : CharRNN générique avec Embedding + RNN/LSTM/GRU + FC
hidden_size = 64, seq_length = 10
Gradient Clipping (max_norm=5.0) pour éviter l'explosion des gradients
BPTT (Backpropagation Through Time) géré par PyTorch
Entraînement : 20 epochs avec LR adaptés :
RNN simple : lr=0.05 (convergence lente, vanishing gradient)
LSTM / GRU : lr=0.005 (portes stabilisant les gradients)
--3.2 Traduction Automatique Seq2Seq

Dataset : 20 paires de phrases Français ↔ Anglais
Vocabulaire : 97 mots avec tokens spéciaux <SOS>, <EOS>, <UNK>
Architecture :
EncoderRNN : Embedding + GRU (hidden=64)
DecoderRNN : Embedding + GRU + Linear(output)
Teacher Forcing (ratio=0.5) + Gradient Clipping
Entraînement : 30 epochs, lr=0.001
Loss finale : 0.1323
Décodage :
Greedy Search : sélection du mot le plus probable à chaque pas
Beam Search (beam_width=3) : exploration de plusieurs hypothèses
Évaluation : Score BLEU-n avec smoothing et Brevity Penalty


Exécution
Ouvrez chaque notebook Jupyter et exécutez les cellules dans l'ordre.

📁 Fichiers Sauvegardés
Modèles MLP : best_mlp_*.pth (poids sauvegardés pour chaque initialisation)
Visualisations CNN : feature maps, courbes de comparaison, tableau CSV des variantes
Visualisations RNN : courbes Loss/Perplexité, résultats de traduction CSV

Auteur
Nom : El aouazi hafsa
Date : Juin 2026

📚 Références
PyTorch Documentation
Fashion-MNIST Dataset
BLEU Score Paper
