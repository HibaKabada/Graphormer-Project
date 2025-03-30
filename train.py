# Tout ce qu'il faut pour faire tourner l'entraînement
import os
import random
from tqdm import tqdm, trange  
import torch
import torch.nn as nn
import torch.optim as optim

# Nos fichiers persos
from data_loader import load_esol_dataset
from graphormer_model import Graphormer
from config import get_config, ExperimentLogger, print_config_table


def initialize_experiment(config):
    """Prépare tout ce qu'il faut pour l'expérience : dossiers, sauvegarde du code..."""
    # On crée le dossier de sortie s'il existe pas
    os.makedirs('outputs', exist_ok=True)
    
    # Un sous-dossier pour cette expérience spécifique
    experiment_dir = os.path.join('outputs', config.exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # On fait une copie de sécurité de tous les fichiers importants
    # Comme ça si on modifie après, on garde la version utilisée
    code_files = ['train.py', 'data_loader.py', 'graphormer_layers.py',
                 'graphormer_model.py', 'config.py']
    for file in code_files:
        if os.path.exists(file):
            os.system(f'cp {file} {experiment_dir}/{file}.backup')


def train_model(config, logger, train_loader, num_node_features, num_edge_features):
    """La boucle principale d'entraînement du modèle."""
    # On choisit si on utilise le GPU ou le CPU
    device = torch.device(f'cuda:{config.gpu_index}' if config.gpu_index >= 0 else 'cpu')
    logger.log(f'On va utiliser : {device}')

    # On crée notre modèle Graphormer et on l'envoie sur le bon device
    model = Graphormer(config, num_node_features, num_edge_features).to(device)
    logger.log(str(model))  # On affiche l'architecture

    # Petite info utile : combien de paramètres à apprendre ?
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f'Nombre de paramètres à entraîner : {total_params:,}')

    # Les outils pour l'optimisation
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)  # Un optimiseur moderne
    criterion = nn.L1Loss(reduction="sum")  # Notre fonction de perte

    logger.log('Lancement de l\'entraînement...')
    # La grande boucle sur toutes les époques
    for epoch in trange(config.num_epochs, desc="Training"):
        model.train()  # Mode entraînement
        epoch_loss = 0.0  # Pour calculer la perte moyenne

        # On parcourt tous les batchs de données
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)  # On envoie les données sur le bon device

            # Étape 1 : Forward pass (calcul des prédictions)
            optimizer.zero_grad()  # On réinitialise les gradients
            predictions = model(batch)  # Forward pass
            loss = criterion(predictions, batch.y)  # Calcul de l'erreur

            # Étape 2 : Backward pass (rétropropagation)
            loss.backward()  # Calcul des gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # On évite les gradients explosifs
            optimizer.step()  # Mise à jour des poids

            epoch_loss += loss.item()  # On accumule la perte

        # À la fin de chaque époque, on affiche la perte moyenne
        avg_loss = epoch_loss / len(train_loader.dataset)
        logger.log(f'Epoque {epoch:03d} | Perte moyenne : {avg_loss:.6f}')

    # Sauvegarde du modèle entraîné
    model_path = os.path.join('outputs', config.exp_name, 'model.pth')
    torch.save(model, model_path)
    logger.log(f'Modèle sauvegardé : {model_path}')


def evaluate_model(config, logger, test_loader):
    """Évalue le modèle sur les données de test."""
    device = torch.device(f'cuda:{config.gpu_index}' if config.gpu_index >= 0 else 'cpu')

    logger.log('\n' + '='*50)
    logger.log('Début de l\'évaluation...')
    logger.log(f'Chargement du modèle : outputs/{config.exp_name}/model.pth')

    # On charge le modèle qu'on vient d'entraîner
    model_path = os.path.join('outputs', config.exp_name, 'model.pth')
    model = torch.load(model_path).to(device)
    model.eval()  # Mode évaluation

    # Initialisation des métriques
    criterion = nn.L1Loss(reduction="sum")
    test_loss = 0.0

    # Pas besoin de calculer les gradients pour l'évaluation
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            predictions = model(batch)
            test_loss += criterion(predictions, batch.y).item()

    # Affichage des résultats
    avg_test_loss = test_loss / len(test_loader.dataset)
    logger.log(f'Perte sur le test : {avg_test_loss:.6f}')
    logger.log('='*50 + '\n')


if __name__ == '__main__':
    # Initialisation de l'expérience
    config = get_config()  # On récupère la configuration
    random.seed(config.seed)  # On fixe les seeds pour la reproductibilité
    torch.manual_seed(config.seed)

    # Préparation des dossiers et logs
    initialize_experiment(config)
    logger = ExperimentLogger(os.path.join('outputs', config.exp_name, 'experiment.log'))
    logger.log(print_config_table(config))  # On log la config

    # Chargement des données
    train_loader, test_loader, num_node_features, num_edge_features = load_esol_dataset(config)

    # Lancement de l'entraînement et de l'évaluation
    train_model(config, logger, train_loader, num_node_features, num_edge_features)
    evaluate_model(config, logger, test_loader)
