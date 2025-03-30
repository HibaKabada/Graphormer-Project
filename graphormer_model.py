from typing import Dict, List, Tuple
import torch
import networkx as nx
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils.convert import to_networkx

from graphormer_layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding

def compute_shortest_paths(graph: nx.Graph, source: int, cutoff: int = None) -> Tuple[Dict, Dict]:
    """Calcule les plus courts chemins depuis un nœud source vers tous les autres.
    
    Args:
        graph: Le graphe NetworkX
        source: Index du nœud de départ
        cutoff: Longueur maximale des chemins à considérer

    Returns:
        Un tuple avec:
            - node_paths: Dictionnaire des chemins (séquences de nœuds)
            - edge_paths: Dictionnaire des chemins (séquences d'arêtes)
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source} not in graph")

    # On crée un mapping des arêtes pour les retrouver facilement
    edges = {edge: i for i, edge in enumerate(graph.edges())}

    level = 0
    next_level = {source: 1}  # On commence par le nœud source
    node_paths = {source: [source]}  # Chemin vers soi-même = [soi-même]
    edge_paths = {source: []}  # Pas d'arête pour aller à soi-même

    # Algorithme BFS (Breadth-First Search)
    while next_level:
        current_level = next_level
        next_level = {}

        # On explore tous les voisins du niveau actuel
        for node in current_level:
            for neighbor in graph[node]:
                if neighbor not in node_paths:  # Si pas encore visité
                    # On construit le chemin en ajoutant le voisin
                    node_paths[neighbor] = node_paths[node] + [neighbor]
                    # On récupère l'index de l'arête entre les 2 derniers nœuds
                    edge_paths[neighbor] = edge_paths[node] + [edges[tuple(node_paths[neighbor][-2:])]]
                    next_level[neighbor] = 1  # À explorer au prochain niveau

        level += 1
        if cutoff is not None and cutoff <= level:  # Si on a atteint la limite
            break

    return node_paths, edge_paths


def all_pairs_shortest_paths(graph: nx.Graph) -> Tuple[Dict, Dict]:
    """Calcule les plus courts chemins entre toutes les paires de nœuds.
    
    Args:
        graph: Le graphe NetworkX

    Returns:
        Un tuple avec:
            - node_paths: Dictionnaire des chemins entre tous les nœuds
            - edge_paths: Dictionnaire des arêtes empruntées
    """
    # On calcule les chemins pour chaque nœud comme source
    paths = {n: compute_shortest_paths(graph, n) for n in graph}
    # On sépare les chemins de nœuds et d'arêtes
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def get_graph_shortest_paths(graph_data: Data) -> Tuple[Dict, Dict]:
    """Calcule les plus courts chemins pour un seul graphe PyG.
    
    Args:
        graph_data: Un objet Data de PyTorch Geometric

    Returns:
        Un tuple avec les chemins (nœuds et arêtes)
    """
    # Conversion en graphe NetworkX + calcul des chemins
    graph = to_networkx(graph_data)
    return all_pairs_shortest_paths(graph)


def get_batched_shortest_paths(batch_data: Data) -> Tuple[Dict, Dict]:
    """Calcule les plus courts chemins pour un batch de graphes.
    
    Args:
        batch_data: Un batch de graphes PyG

    Returns:
        Un tuple avec les chemins pour tout le batch
    """
    # On sépare le batch en graphes individuels
    graph_list = [to_networkx(graph) for graph in batch_data.to_data_list()]

    # On rend les indices de nœuds uniques dans tout le batch
    relabeled_graphs = []
    node_offset = 0
    for graph in graph_list:
        num_nodes = graph.number_of_nodes()
        # On décale les indices pour éviter les collisions
        mapping = {i: i + node_offset for i in range(num_nodes)}
        relabeled_graphs.append(nx.relabel_nodes(graph, mapping))
        node_offset += num_nodes  # On incrémente le décalage

    # On calcule les chemins pour chaque graphe et on combine
    all_node_paths = {}
    all_edge_paths = {}

    for graph in relabeled_graphs:
        node_paths, edge_paths = all_pairs_shortest_paths(graph)
        all_node_paths.update(node_paths)
        all_edge_paths.update(edge_paths)

    return all_node_paths, all_edge_paths


class Graphormer(nn.Module):
    """Modèle Graphormer pour l'apprentissage de représentations de graphes."""

    def __init__(self, config, num_node_features: int, num_edge_features: int):
        """
        Args:
            config: Configuration du modèle
            num_node_features: Nombre de features par nœud en entrée
            num_edge_features: Nombre de features par arête en entrée
        """
        super().__init__()

        # On stocke la configuration
        self.num_layers = config.num_layers
        self.hidden_dim = config.node_dim
        self.edge_hidden_dim = config.edge_dim
        self.num_heads = config.num_heads
        self.output_dim = config.output_dim

        # Couches de projection pour les features d'entrée
        self.node_input_proj = nn.Linear(num_node_features, self.hidden_dim)
        self.edge_input_proj = nn.Linear(num_edge_features, self.edge_hidden_dim)

        # Modules d'encodage
        self.centrality_encoder = CentralityEncoding(
            max_in_degree=config.max_in_degree,
            max_out_degree=config.max_out_degree,
            hidden_dim=self.hidden_dim
        )

        self.spatial_encoder = SpatialEncoding(
            max_path_distance=config.max_path_distance
        )

        # Couches Transformer empilées
        self.encoder_layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_feature_dim=self.hidden_dim,
                edge_feature_dim=self.edge_hidden_dim,
                num_heads=self.num_heads,
                max_path_distance=config.max_path_distance
            ) for _ in range(self.num_layers)
        ])

        # Projection finale
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, graph_data: Data) -> torch.Tensor:
        """Passe avant du modèle complet.
        
        Args:
            graph_data: Un graphe ou batch de graphes

        Returns:
            Les prédictions au niveau graphe
        """
        # On récupère les features d'entrée
        node_features = graph_data.x.float()
        edge_index = graph_data.edge_index.long()
        edge_features = graph_data.edge_attr.float()

        # On vérifie si c'est un batch ou un seul graphe
        batch_ptr = getattr(graph_data, 'ptr', None)

        # Calcul des plus courts chemins
        if isinstance(graph_data, Data):  # Un seul graphe
            node_paths, edge_paths = get_graph_shortest_paths(graph_data)
        else:  # Batch de graphes
            node_paths, edge_paths = get_batched_shortest_paths(graph_data)

        # Projection des features
        node_features = self.node_input_proj(node_features)
        edge_features = self.edge_input_proj(edge_features)

        # Ajout de l'encodage de centralité
        node_features = self.centrality_encoder(node_features, edge_index)

        # Calcul de l'encodage spatial
        spatial_encoding = self.spatial_encoder(node_features, node_paths)

        # Application des couches Transformer
        for layer in self.encoder_layers:
            node_features = layer(
                node_features=node_features,
                edge_features=edge_features,
                spatial_encoding=spatial_encoding,
                edge_paths=edge_paths,
                batch_ptr=batch_ptr
            )

        # Représentation globale du graphe (moyenne des nœuds)
        node_features = self.output_proj(node_features)
        graph_representation = global_mean_pool(node_features, graph_data.batch)

        return graph_representation
