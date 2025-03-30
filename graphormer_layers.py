from typing import Tuple
import torch
from torch import nn
from torch_geometric.utils import degree

class CentralityEncoding(nn.Module):
    """Donne plus d'importance aux nœuds centraux du graphe en fonction de leur connectivité."""

    def __init__(self, max_in_degree: int, max_out_degree: int, hidden_dim: int):
        """
        Args:
            max_in_degree: Nombre max de connexions entrantes qu'un nœud peut avoir
            max_out_degree: Nombre max de connexions sortantes
            hidden_dim: Taille des vecteurs de caractéristiques
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree

        # Chaque type de degré aura sa propre représentation vectorielle apprenable
        self.in_degree_embedding = nn.Parameter(torch.randn((max_in_degree, hidden_dim)))
        self.out_degree_embedding = nn.Parameter(torch.randn((max_out_degree, hidden_dim)))

    def forward(self, node_features: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """Améliore les caractéristiques des nœuds avec leur importance dans le réseau."""
        num_nodes = node_features.shape[0]

        # Calcul du nombre de connexions entrantes/sortantes pour chaque nœud
        in_degree = self._clip_degrees(
            degree(index=edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1  # -1 car on indexe à partir de 0
        )
        out_degree = self._clip_degrees(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1
        )

        # Ajout des informations de centralité aux caractéristiques existantes
        node_features += self.in_degree_embedding[in_degree] + self.out_degree_embedding[out_degree]

        return node_features

    def _clip_degrees(self, degrees: torch.Tensor, max_value: int) -> torch.Tensor:
        """Écrête les valeurs trop élevées pour rester dans les limites définies."""
        degrees[degrees > max_value] = max_value
        return degrees


class SpatialEncoding(nn.Module):
    """Prend en compte la distance entre les nœuds dans le graphe."""

    def __init__(self, max_path_distance: int):
        """
        Args:
            max_path_distance: Distance maximale entre deux nœuds à considérer
        """
        super().__init__()
        self.max_path_distance = max_path_distance
        # Chaque distance possible a un poids spécifique à apprendre
        self.distance_weights = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, node_features: torch.Tensor, shortest_paths: dict) -> torch.Tensor:
        """Crée une matrice qui indique comment les nœuds sont éloignés les uns des autres."""
        device = next(self.parameters()).device
        num_nodes = node_features.shape[0]
        spatial_encoding_matrix = torch.zeros((num_nodes, num_nodes), device=device)

        # Pour chaque paire de nœuds connectés
        for src in shortest_paths:
            for dst in shortest_paths[src]:
                # On prend la distance ou la limite maximale si le chemin est trop long
                path_length = min(len(shortest_paths[src][dst]), self.max_path_distance)
                # On attribue un poids en fonction de cette distance
                spatial_encoding_matrix[src][dst] = self.distance_weights[path_length - 1]

        return spatial_encoding_matrix


class EdgeEncoding(nn.Module):
    """Tient compte des caractéristiques des arêtes empruntées dans les chemins."""

    def __init__(self, edge_feature_dim: int, max_path_distance: int):
        """
        Args:
            edge_feature_dim: Dimension des caractéristiques des arêtes
            max_path_distance: Longueur maximale des chemins à considérer
        """
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.max_path_distance = max_path_distance
        # Représentations apprenables pour chaque position possible dans un chemin
        self.path_embeddings = nn.Parameter(torch.randn(self.max_path_distance, self.edge_feature_dim))

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
               edge_paths: dict) -> torch.Tensor:
        """Crée une matrice qui encode les caractéristiques des arêtes sur les chemins."""
        device = next(self.parameters()).device
        num_nodes = node_features.shape[0]
        edge_encoding_matrix = torch.zeros((num_nodes, num_nodes), device=device)

        # Pour chaque chemin entre nœuds
        for src in edge_paths:
            for dst in edge_paths[src]:
                # On prend les arêtes du chemin (tronqué si trop long)
                path_indices = edge_paths[src][dst][:self.max_path_distance]
                weight_indices = torch.arange(len(path_indices), device=device)
                # On combine les caractéristiques des arêtes avec nos embeddings
                edge_encoding_matrix[src][dst] = self._weighted_edge_sum(
                    self.path_embeddings[weight_indices],
                    edge_features[path_indices]
                ).mean()  # Moyenne le long du chemin

        return torch.nan_to_num(edge_encoding_matrix)  # Évite les NaN

    def _weighted_edge_sum(self, embeddings: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """Combine linéairement les embeddings et les caractéristiques des arêtes."""
        return (embeddings * edge_features).sum(dim=1)


class GraphormerAttentionHead(nn.Module):
    """Une tête d'attention qui tient compte de la structure du graphe."""

    def __init__(self, node_feature_dim: int, query_dim: int, key_dim: int,
                 edge_feature_dim: int, max_path_distance: int):
        """
        Args:
            node_feature_dim: Dimension des caractéristiques des nœuds
            query_dim: Dimension pour les requêtes
            key_dim: Dimension pour les clés/valeurs
            edge_feature_dim: Dimension des caractéristiques des arêtes
            max_path_distance: Distance maximale entre nœuds
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_feature_dim, max_path_distance)

        # Transformations linéaires pour les requêtes, clés et valeurs
        self.query_proj = nn.Linear(node_feature_dim, query_dim)
        self.key_proj = nn.Linear(node_feature_dim, key_dim)
        self.value_proj = nn.Linear(node_feature_dim, key_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                spatial_encoding: torch.Tensor, edge_paths: dict,
                batch_ptr: torch.Tensor = None) -> torch.Tensor:
        """Calcule l'attention en intégrant les informations structurelles du graphe."""
        device = next(self.parameters()).device
        num_nodes = node_features.shape[0]

        # Masque pour ne considérer que les nœuds pertinents
        attention_mask = torch.full((num_nodes, num_nodes), -1e6, device=device)
        output_mask = torch.zeros((num_nodes, num_nodes), device=device)

        if batch_ptr is None:  # Cas d'un seul graphe
            attention_mask.fill_(1)
            output_mask.fill_(1)
        else:  # Cas d'un batch de graphes
            for i in range(len(batch_ptr) - 1):
                start, end = batch_ptr[i], batch_ptr[i + 1]
                attention_mask[start:end, start:end] = 1  # Masque intra-graphe
                output_mask[start:end, start:end] = 1

        # Transformation des caractéristiques
        queries = self.query_proj(node_features)  # Ce qu'on cherche
        keys = self.key_proj(node_features)       # Ce qu'on compare
        values = self.value_proj(node_features)   # Ce qu'on retourne

        # Calcul des scores d'attention
        edge_encodings = self.edge_encoding(node_features, edge_features, edge_paths)
        attention_scores = self._compute_attention_scores(keys, queries, batch_ptr)

        # Combinaison des différents types d'information
        attention_scores = (attention_scores + spatial_encoding + edge_encodings) * attention_mask
        attention_weights = torch.softmax(attention_scores, dim=-1) * output_mask

        # Mise à jour des caractéristiques
        return attention_weights @ values

    def _compute_attention_scores(self, keys: torch.Tensor, queries: torch.Tensor,
                                 batch_ptr: torch.Tensor = None) -> torch.Tensor:
        """Calcule les affinités entre requêtes et clés."""
        if batch_ptr is None:  # Cas simple d'un seul graphe
            return queries @ keys.T / (queries.size(-1) ** 0.5  # Produit scalaire normalisé

        # Version optimisée pour les batchs
        attention_scores = torch.zeros((queries.shape[0], queries.shape[0]), device=keys.device)
        for i in range(len(batch_ptr) - 1):
            start, end = batch_ptr[i], batch_ptr[i + 1]
            batch_queries = queries[start:end]
            batch_keys = keys[start:end]
            attention_scores[start:end, start:end] = (
                batch_queries @ batch_keys.T / (queries.size(-1) ** 0.5)
        return attention_scores


class GraphormerMultiHeadAttention(nn.Module):
    """Attention multi-têtes pour capturer différents types de relations."""

    def __init__(self, num_heads: int, node_feature_dim: int, query_dim: int,
                 key_dim: int, edge_feature_dim: int, max_path_distance: int):
        """
        Args:
            num_heads: Nombre de têtes d'attention en parallèle
            node_feature_dim: Dimension des caractéristiques des nœuds
            query_dim: Dimension pour les requêtes
            key_dim: Dimension pour les clés/valeurs
            edge_feature_dim: Dimension des caractéristiques des arêtes
            max_path_distance: Distance maximale entre nœuds
        """
        super().__init__()
        # Plusieurs têtes pour diversifier l'apprentissage
        self.attention_heads = nn.ModuleList([
            GraphormerAttentionHead(
                node_feature_dim=node_feature_dim,
                query_dim=query_dim,
                key_dim=key_dim,
                edge_feature_dim=edge_feature_dim,
                max_path_distance=max_path_distance
            ) for _ in range(num_heads)
        ])
        # Combinaison des sorties des différentes têtes
        self.output_proj = nn.Linear(num_heads * key_dim, node_feature_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                spatial_encoding: torch.Tensor, edge_paths: dict,
                batch_ptr: torch.Tensor) -> torch.Tensor:
        """Applique l'attention multi-têtes et combine les résultats."""
        # Chaque tête calcule sa propre version
        head_outputs = [
            head(node_features, edge_features, spatial_encoding, edge_paths, batch_ptr)
            for head in self.attention_heads
        ]
        # On fusionne les résultats
        return self.output_proj(torch.cat(head_outputs, dim=-1))


class GraphormerEncoderLayer(nn.Module):
    """Une couche complète du transformeur adaptée aux graphes."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int,
                 num_heads: int, max_path_distance: int):
        """
        Args:
            node_feature_dim: Dimension des caractéristiques des nœuds
            edge_feature_dim: Dimension des caractéristiques des arêtes
            num_heads: Nombre de têtes d'attention
            max_path_distance: Distance maximale entre nœuds
        """
        super().__init__()

        # Mécanisme d'attention multi-têtes
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            node_feature_dim=node_feature_dim,
            query_dim=node_feature_dim,
            key_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            max_path_distance=max_path_distance,
        )

        # Normalisation et réseau feed-forward
        self.pre_norm = nn.LayerNorm(node_feature_dim)
        self.post_norm = nn.LayerNorm(node_feature_dim)
        self.ffn = nn.Linear(node_feature_dim, node_feature_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                spatial_encoding: torch.Tensor, edge_paths: dict,
                batch_ptr: torch.Tensor) -> torch.Tensor:
        """Une passe avant complète de la couche d'encodage."""
        # Attention avec saut de connexion
        attended_features = self.attention(
            self.pre_norm(node_features), edge_features, spatial_encoding,
            edge_paths, batch_ptr
        ) + node_features  # Connexion résiduelle

        # Réseau feed-forward avec saut de connexion
        return self.ffn(self.post_norm(attended_features)) + attended_features
