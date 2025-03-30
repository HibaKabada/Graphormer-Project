import os
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def load_esol_dataset(config):
   """Charge et sépare le dataset ESOL pour la prédiction de propriétés moléculaires.

   Args:
        config: Objet de configuration avec les paramètres

    Returns:
        train_loader: DataLoader pour l'entraînement
        test_loader: DataLoader pour les tests
        num_node_features: Nombre de features par nœud
        num_edge_features: Nombre de features par arête
    """
   # On s'assure que le dossier Data/MoleculeNet existe (sinon on le crée)
    os.makedirs('Data/MoleculeNet', exist_ok=True)

    # Chargement du dataset ESOL - c'est des molécules avec leurs propriétés solubilité
    dataset = MoleculeNet(root='Data/MoleculeNet', name='ESOL')

     # Séparation train/test (80% pour l'entraînement, 20% pour les tests)
    num_samples = len(dataset)  #Nombre total de molécules dans le dataset
    train_size = int(0.8 * num_samples) # 80% pour l'entraînement

    train_dataset = dataset[:train_size]  # Les premières 80% des données
    test_dataset = dataset[train_size:] # Les 20% restantes

     # Création des DataLoaders :
    # - Pour l'entraînement : on mélange les données (shuffle=True)
    # - Pour les tests : pas besoin de mélanger
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    # On récupère la dimension des features :
    # - x.shape[1] donne le nombre de features par nœud (atome)
    # - edge_attr.shape[1] donne le nombre de features par arête (liaison)
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0

    return train_loader, test_loader, num_node_features, num_edge_features
