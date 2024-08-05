import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Linear, GATv2Conv, SAGEConv


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=None, flatten=True):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.flatten = flatten
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x):
        z = self.lin1(x).relu()
        z = self.dropout(z)
        z = self.lin2(z).relu()
        z = self.dropout(z)
        z = self.lin3(z)
        if self.flatten:
            return z.view(-1)
        else:
            return z


class HomoGATEncoder(nn.Module):
    """
    Edge regression from the node embedding from a directed weighted graph.
    x = ... # Node feature matrix: [num_nodes, num_features]
    edge_index = ... # Edge indices: [2, num_edges]
    edge_value = ... # Edge attribute (OD volume): [num_edges, 1]
    """

    def __init__(self,
                 hidden_channels,
                 out_channels,
                 heads, dropout,
                 num_layers=3,
                 layer_type='gat',
                 analyze_mode=False):
        super().__init__()

        self.num_layers = num_layers
        self.layer_type = layer_type
        self.analyze_mode = analyze_mode
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.layer_type == 'gat':
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=1))
            self.lins.append(Linear(hidden_channels, hidden_channels * heads))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1))
            self.lins.append(Linear(hidden_channels * heads, hidden_channels * heads))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads,
                          edge_dim=1, concat=False))
            self.lins.append(Linear(hidden_channels * heads, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.layer_type == 'sage':
            for i in range(num_layers):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.lins.append(Linear(hidden_channels, hidden_channels))
                self.norms.append(nn.BatchNorm1d(hidden_channels))
        else:
            raise NotImplementedError('Wrong layer type!')

        self.in_emb_encoder = nn.Linear(hidden_channels, out_channels)
        self.out_emb_encoder = nn.Linear(hidden_channels, out_channels)
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.in_emb_encoder.reset_parameters()
        self.out_emb_encoder.reset_parameters()

    def forward(self, edge_index, node_feature, edge_weight):
        x = node_feature

        for i in range(self.num_layers - 1):
            if self.layer_type == 'gat':
                x = self.norms[i](self.convs[i](x, edge_index, edge_weight) + self.lins[i](x))
            elif self.layer_type == 'sage':
                x = self.norms[i](self.convs[i](x, edge_index) + self.lins[i](x))
            x = self.dropout(F.leaky_relu(x))

        if self.layer_type == 'gat':
            x = self.norms[-1](self.convs[-1](x, edge_index, edge_weight) + self.lins[-1](x))
        elif self.layer_type == 'sage':
            x = self.norms[-1](self.convs[-1](x, edge_index) + self.lins[-1](x))

        in_embed = self.in_emb_encoder(x)
        out_embed = self.out_emb_encoder(x)
        return in_embed, out_embed, x


class SpatialGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads, dropout, layer_type='gat', analyze_mode=False):
        super().__init__()
        self.projection = MLP(in_channels, hidden_channels, hidden_channels, dropout=dropout, flatten=False)
        self.encoder = HomoGATEncoder(hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      heads=heads,
                                      dropout=dropout,
                                      layer_type=layer_type)
        self.decoder = MLP(out_channels * 2, hidden_channels, 1, dropout=dropout)
        self.analyze_mode = analyze_mode
        self.reset_parameters()

    def forward(self, node_feature, edge_index, edge_weight, edge_label_index):
        node_feature = self.projection(node_feature)
        in_embed, out_embed, general_embed = self.encoder.forward(edge_index, node_feature, edge_weight)
        edge_emb = torch.concat([in_embed[edge_label_index[0]], out_embed[edge_label_index[1]]], dim=-1)
        volume = self.decoder(edge_emb).reshape([-1, 1])

        if self.analyze_mode:
            return volume
        else:
            return in_embed, out_embed, volume, general_embed

    def reset_parameters(self):
        self.projection.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
