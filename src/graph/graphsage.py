"""
GraphSAGE model for link prediction.

Preserved from the legacy implementation with minimal modernization:
- GraphSAGE: two-layer SAGEConv encoder (configurable aggregator)
- MLPPredictor: learned edge scorer via concatenated node embeddings
- ScorePredictor: dot-product edge scorer (lightweight alternative)
- Model: combined encoder + predictor

Changes from legacy:
- Replaced deprecated NodeDataLoader with DataLoader
- Cleaned up device handling
- Added type hints
- Kept all GraphSAGE logic and architecture intact
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Generates node embeddings by sampling and aggregating features from
    each node's local neighborhood. Supports mean, lstm, gcn, and pool aggregators.
    """

    def __init__(self, in_feats: int, hid_feats: int, out_feats: int,
                 aggregator_type: str = "mean"):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.n_layers = 2

        self.conv1 = SAGEConv(
            in_feats=in_feats,
            out_feats=hid_feats,
            aggregator_type=aggregator_type,
        )
        self.conv2 = SAGEConv(
            in_feats=hid_feats,
            out_feats=out_feats,
            aggregator_type=aggregator_type,
        )

    def forward(self, blocks, inputs):
        """Mini-batch forward through message-passing blocks."""
        h = self.conv1(blocks[0], inputs)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h

    def inference(self, g, x, batch_size, device="cpu"):
        """
        Full-graph inference: compute embeddings layer by layer.

        Used after training to generate embeddings for all nodes.
        """
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(
                g.number_of_nodes(),
                self.hid_feats if l != self.n_layers - 1 else self.out_feats,
            )
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = F.relu(layer(block, (h, h_dst)))
                y[output_nodes] = h.cpu()

            x = y
        return y


class ScorePredictor(nn.Module):
    """Dot-product edge score predictor."""

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v("x", "x", "score"))
            return edge_subgraph.edata["score"]


class MLPPredictor(nn.Module):
    """
    MLP edge score predictor.

    Concatenates source and destination node embeddings, applies a linear
    layer to produce a scalar score for edge existence likelihood.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.W = nn.Linear(in_features * 2, 1)

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class Model(nn.Module):
    """
    Combined GraphSAGE encoder + MLP link predictor.

    Forward pass: encode node features via GraphSAGE blocks, then score
    positive and negative edge subgraphs via MLPPredictor.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int, num_classes: int,
                 aggregator_type: str = "mean"):
        super().__init__()
        self.gcn = GraphSAGE(in_features, hidden_features, out_features,
                             aggregator_type)
        self.pred = MLPPredictor(out_features)

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score

    def get_embeddings(self, g, x, batch_size, device,
                       provide_prediction=False):
        if provide_prediction:
            return self.pred(g, x)
        else:
            return self.gcn.inference(g, x, batch_size, device)
