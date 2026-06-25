"""Model definitions for baseline and graph representation learning runs.

Expected contents over time:

- a local-discrepancy scoring baseline;
- coordinate-wise graph models following the architecture in ``spec.md``.

Keep early implementations pragmatic. If this file becomes hard to navigate, it
can be split later.
"""

from __future__ import annotations

import torch
from torch import nn

from grl.graphs import GraphKind, KnnMetric, build_tensor_graph, tensor_coordinate_ranks


def make_encoder_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build the two-layer encoder MLP used for node and edge features."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


def make_projection_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build the message/update MLP used inside graph layers."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class FlatSetMlpPointScorer(nn.Module):
    """Fixed-shape scorer that is sensitive to point and coordinate order."""

    def __init__(self, n: int, d: int, hidden_dim: int = 128):
        super().__init__()
        if n <= 0:
            raise ValueError("n must be positive")
        if d <= 0:
            raise ValueError("d must be positive")
        self.n = n
        self.d = d
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(n * d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
        )
        self.last_coordinate_logits: torch.Tensor | None = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if tuple(points.shape[1:]) != (self.n, self.d):
            raise ValueError(
                f"expected points with shape (*, {self.n}, {self.d}), "
                f"got {tuple(points.shape)}"
            )
        self.last_coordinate_logits = None
        return self.net(points.reshape(points.shape[0], self.n * self.d))


class CoordinateKnnGraphPointScorer(nn.Module):
    """Coordinate-wise graph scorer.

    The graph is rebuilt from the current points on every forward pass. In the
    default shared-coordinate setting, node, edge, message, update, and score
    functions are shared over coordinate indices, so the model can transfer
    across dimensions. With ``coordinate_shared=False`` the same architecture is
    used with coordinate-index-specific functions, which intentionally removes
    coordinate permutation equivariance and dimension transfer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        k: int = 8,
        metric: KnnMetric = "euclidean",
        graph_kind: GraphKind | None = None,
        coordinate_shared: bool = True,
        input_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        resolved_graph_kind = graph_kind or f"knn_{metric}"
        if resolved_graph_kind not in {
            "knn_euclidean",
            "knn_linf",
            "rank_knn_euclidean",
            "rank_knn_linf",
            "rank_adjacency",
        }:
            raise ValueError(f"unknown graph_kind: {resolved_graph_kind}")
        if num_layers > 0 and resolved_graph_kind != "rank_adjacency" and k <= 0:
            raise ValueError("k must be positive for kNN graphs")
        if num_layers == 0 and k < 0:
            raise ValueError("k must be non-negative")
        if metric not in {"euclidean", "linf"}:
            raise ValueError("metric must be 'euclidean' or 'linf'")
        if not coordinate_shared and input_dim is None:
            raise ValueError("input_dim is required when coordinate_shared is false")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.metric = metric
        self.graph_kind = resolved_graph_kind
        self.coordinate_shared = coordinate_shared
        self.input_dim = input_dim
        self.eps = eps

        if coordinate_shared:
            self.node_encoder = make_encoder_mlp(4, hidden_dim)
            self.edge_encoder = make_encoder_mlp(6, hidden_dim) if num_layers > 0 else None
            self.message_mlps = nn.ModuleList(
                [make_projection_mlp(5 * hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.update_mlps = nn.ModuleList(
                [make_projection_mlp(6 * hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            )
            self.coordinate_head = nn.Sequential(
                nn.Linear(5 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            d = int(input_dim)
            self.node_encoders = nn.ModuleList([make_encoder_mlp(4, hidden_dim) for _ in range(d)])
            self.edge_encoders = (
                nn.ModuleList([make_encoder_mlp(6, hidden_dim) for _ in range(d)])
                if num_layers > 0
                else None
            )
            self.message_mlps_by_coord = nn.ModuleList(
                [
                    nn.ModuleList(
                        [make_projection_mlp(5 * hidden_dim, hidden_dim) for _ in range(d)]
                    )
                    for _ in range(num_layers)
                ]
            )
            self.update_mlps_by_coord = nn.ModuleList(
                [
                    nn.ModuleList(
                        [make_projection_mlp(6 * hidden_dim, hidden_dim) for _ in range(d)]
                    )
                    for _ in range(num_layers)
                ]
            )
            self.layer_norms_by_coord = nn.ModuleList(
                [nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(d)]) for _ in range(num_layers)]
            )
            self.coordinate_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(5 * hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(d)
                ]
            )
        self.last_coordinate_logits: torch.Tensor | None = None

    @property
    def uses_knn(self) -> bool:
        return self.graph_kind != "rank_adjacency"

    def coordinate_features(self, points: torch.Tensor) -> torch.Tensor:
        """Build shared per-coordinate node features for ``(batch, n, d)`` points."""
        ranks = tensor_coordinate_ranks(points)
        clipped = points.clamp(self.eps, 1.0 - self.eps)
        return torch.stack([points, ranks, torch.log(clipped), torch.log1p(-clipped)], dim=-1)

    def encode_coordinates(
        self,
        values: torch.Tensor,
        modules: nn.ModuleList,
    ) -> torch.Tensor:
        """Apply one module per coordinate to a tensor with a coordinate axis."""
        if self.input_dim is None or values.shape[-2] != self.input_dim:
            raise ValueError(
                f"expected coordinate dimension {self.input_dim}, got {values.shape[-2]}"
            )
        encoded = [module(values[..., coord, :]) for coord, module in enumerate(modules)]
        return torch.stack(encoded, dim=-2)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return one point logit for each input point."""
        batch_size, n, d = points.shape
        if not self.coordinate_shared and d != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {d}")

        node_features = self.coordinate_features(points)
        if self.coordinate_shared:
            h = self.node_encoder(node_features)
        else:
            h = self.encode_coordinates(node_features, self.node_encoders)

        if self.num_layers > 0:
            graph = build_tensor_graph(points, kind=self.graph_kind, k=self.k)
            if self.coordinate_shared:
                if self.edge_encoder is None:
                    raise RuntimeError("edge_encoder is required when num_layers is positive")
                edge_encoded = self.edge_encoder(graph.edge_attr)
            else:
                if self.edge_encoders is None:
                    raise RuntimeError("edge_encoders are required when num_layers is positive")
                edge_encoded = self.encode_coordinates(graph.edge_attr, self.edge_encoders)

            batch_index = torch.arange(batch_size, device=points.device)[:, None, None]
            neighbor_count = graph.neighbors.shape[2]
            edge_mask_float = graph.edge_mask[:, :, :, None, None].float()
            source_h = h.unsqueeze(2).expand(batch_size, n, neighbor_count, d, self.hidden_dim)

            layer_count = range(self.num_layers)
            for layer_index in layer_count:
                neighbor_h = h[batch_index, graph.neighbors]
                edge_mean = edge_encoded.mean(dim=3, keepdim=True).expand_as(edge_encoded)
                edge_max = edge_encoded.max(dim=3, keepdim=True).values.expand_as(edge_encoded)
                message_inputs = torch.cat(
                    [source_h, neighbor_h, edge_encoded, edge_mean, edge_max],
                    dim=-1,
                )
                if self.coordinate_shared:
                    messages = self.message_mlps[layer_index](message_inputs)
                else:
                    messages = self.encode_coordinates(
                        message_inputs,
                        self.message_mlps_by_coord[layer_index],
                    )
                messages = (messages * edge_mask_float).sum(dim=2)

                coord_mean = h.mean(dim=2, keepdim=True).expand_as(h)
                coord_max = h.max(dim=2, keepdim=True).values.expand_as(h)
                global_mean = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
                global_max = h.amax(dim=(1, 2), keepdim=True).expand_as(h)
                update_inputs = torch.cat(
                    [h, messages, coord_mean, coord_max, global_mean, global_max],
                    dim=-1,
                )
                if self.coordinate_shared:
                    update = self.update_mlps[layer_index](update_inputs)
                    h = self.layer_norms[layer_index](h + update)
                else:
                    update = self.encode_coordinates(
                        update_inputs,
                        self.update_mlps_by_coord[layer_index],
                    )
                    h = self.encode_coordinates(
                        h + update,
                        self.layer_norms_by_coord[layer_index],
                    )
                source_h = h.unsqueeze(2).expand(batch_size, n, neighbor_count, d, self.hidden_dim)

        coord_mean = h.mean(dim=2, keepdim=True).expand_as(h)
        coord_max = h.max(dim=2, keepdim=True).values.expand_as(h)
        global_mean = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
        global_max = h.amax(dim=(1, 2), keepdim=True).expand_as(h)
        head_inputs = torch.cat([h, coord_mean, coord_max, global_mean, global_max], dim=-1)
        if self.coordinate_shared:
            coordinate_logits = self.coordinate_head(head_inputs).squeeze(-1)
        else:
            coordinate_logits = self.encode_coordinates(head_inputs, self.coordinate_heads).squeeze(-1)
        self.last_coordinate_logits = coordinate_logits

        coordinate_probs = torch.sigmoid(coordinate_logits)
        point_probs = 1.0 - torch.prod(1.0 - coordinate_probs.clamp(max=1.0 - self.eps), dim=-1)
        return torch.logit(point_probs.clamp(self.eps, 1.0 - self.eps))
