import argparse
from texttable import Texttable


def get_config():
     """Parse les arguments en ligne de commande et retourne un objet de configuration."""
    # Création du parser pour les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Graphormer for Molecular Property Prediction')

    # ------------------- Configuration générale -------------------
    # --exp_name: Nom de l'expérience pour le suivi
    parser.add_argument('--exp_name', type=str, default='Graphormer_ESOL',
                       help='Experiment name')
      # --seed: Graine aléatoire pour la reproductibilité(42 est un choix courant)
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # ------------------- Configuration matérielle -------------------
    #--gpu_index: Sélection du GPU à utiliser (-1 pour CPU)
    parser.add_argument('--gpu_index', type=int, default=0,
                       help='GPU index (use -1 for CPU)')

     # ------------------- Gestion des données -------------------
    # Taille des batchs (32 est un bon compromis entre vitesse et stabilité)
    parser.add_argument('--train_batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32,
                       help='Testing batch size')

    # ------------------- Paramètres d'entraînement -------------------
    # Nombre d'époques (100 est raisonnable pour beaucoup de cas)
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    # Taux d'apprentissage (5e-4 est un bon point de départ pour Adam/Transformer)
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')

    # ------------------- Architecture du modèle -------------------
    # Nombre de couches du Graphormer (3 est suffisant pour des graphes moyens)
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of Graphormer layers')
    # Dimensions des features (128 est courant pour des graphes moléculaires)
    parser.add_argument('--node_dim', type=int, default=128,
                       help='Hidden dimension for node features')
    parser.add_argument('--edge_dim', type=int, default=128,
                       help='Hidden dimension for edge features')
    # Nombre de têtes d'attention (8 est standard pour les Transformers)
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    # Dimension de sortie (1 pour la régression, >1 pour classification)
    parser.add_argument('--output_dim', type=int, default=1,
                       help='Output dimension')

    # ------------------- Encodages structurels -------------------
    # Paramètres pour les encodages de degré et distance (5 est empirique)
    parser.add_argument('--max_in_degree', type=int, default=5,
                       help='Maximum in-degree for centrality encoding')
    parser.add_argument('--max_out_degree', type=int, default=5,
                       help='Maximum out-degree for centrality encoding')
    parser.add_argument('--max_path_distance', type=int, default=5,
                       help='Maximum path distance for spatial encoding')

    return parser.parse_args() # Retourne la configuration parsée


class ExperimentLogger:
    """Logger pour suivre les expériences (console + fichier)."""
# Cette classe permet de logger les infos importantes pendant l'exécution.
    def __init__(self, log_file):
        """Initialise le logger avec un fichier de sortie."""
         # Ouvre le fichier en mode 'append' pour ne pas écraser les logs existants
        self.log_file = open(log_file, 'a') # 'a' = mode append

    def log(self, message):
          """Log un message dans la console et dans le fichier."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush() # Force l'écriture immédiate (évite les pertes si crash)

    def close(self):
        """Close log file."""
        self.log_file.close() # Important pour éviter les corruptions


def print_config_table(config):
    """Affiche la configuration sous forme de tableau formaté."""
     # Convertit la config en dictionnaire pour un accès facile
    config_dict = vars(config)
    # Initialise le tableau (Texttable pour un affichage propre)
    table = Texttable()
    table.set_cols_dtype(['t', 't']) #Colonnes de type texte
    rows = [["Parameter", "Value"]]  # En-tête du tableau
    # Ajoute chaque paramètre trié par nom
    for key in sorted(config_dict.keys()):
        # Formatage du nom (ex: 'num_layers' → 'Num Layers')
        name = key.replace('_', ' ').title()
        value = str(config_dict[key]) # Conversion en string pour affichage
        rows.append([name, value])

    table.add_rows(rows) # Ajoute toutes les lignes
    return table.draw()  # Génère le tableau ASCII
