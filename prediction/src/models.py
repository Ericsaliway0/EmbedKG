import json
import networkx as nx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import itertools
import scipy.sparse as sp
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from typing import Callable, Optional, Tuple, Union
from dgl.nn import SAGEConv, GATConv, GraphConv, GINConv, ChebConv

class EGCN_CPU(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN)

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(EGCN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.epsilon = epsilon

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

        # Fully Connected Layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass with Chebyshev polynomial expansion and early stopping.
        """
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        # Early stopping for Chebyshev expansion
        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        # Residual connection
        x_res = x
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class EGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Efficient Graph Convolutional Network (EGCN)

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        - device: Device to run the model on ('cpu' or 'cuda')
        """
        super(EGCN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.epsilon = epsilon
        self.device = device  # Store device information

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k).to(device)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k).to(device)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k).to(device)

        # Fully Connected Layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        ).to(device)

        # Batch Normalization
        self.norm = nn.BatchNorm1d(hidden_feats).to(device)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        """
        Forward pass with Chebyshev polynomial expansion and early stopping.
        """
        g = g.to(self.device)  # Ensure graph is on the correct device
        features = features.to(self.device)  # Ensure features are on the correct device

        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        # Early stopping for Chebyshev expansion
        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break
            prev_x = x_new.clone()

        # Residual connection
        x_res = x
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class EGCN_MLP(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.3):
        super(EGCN_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        x = self.dropout(features)
        return self.mlp(x)

class EGCN_NoStop(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3):
        super(EGCN_NoStop, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.norm = nn.BatchNorm1d(hidden_feats)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        # Full k iterations without stopping
        for _ in range(1, self.k):
            x = F.relu(self.cheb2(g, x))

        x_res = x
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class EGCN_Attn(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4, dropout=0.3):
        super(EGCN_Attn, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads)
        self.gat2 = GATConv(hidden_feats * heads, hidden_feats, heads)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.norm = nn.BatchNorm1d(hidden_feats * heads)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        x = F.elu(self.gat1(g, features))
        x = self.norm(x)
        x = F.elu(self.gat2(g, x))
        x = self.dropout_layer(x)

        return self.mlp(x)

class EGCN_Light(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=2, dropout=0.3):
        super(EGCN_Light, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, out_feats, k)

        self.norm = nn.BatchNorm1d(hidden_feats)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)
        x = self.dropout_layer(x)
        return self.cheb2(g, x)

class EGCN_ResOnly(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.3):
        super(EGCN_ResOnly, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.norm = nn.BatchNorm1d(hidden_feats)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, g, features):
        x = self.norm(features)
        x_res = x.clone()
        x = self.dropout_layer(x) + x_res

        return self.mlp(x)

class GATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, Tuple[int, int]],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True) -> None:
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation
        
        # Add normalization layer
        self.norm = nn.BatchNorm1d(num_heads * out_feats)

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        """Set the flag to allow zero in-degree for the graph."""
        self._allow_zero_in_degree = set_value

    def forward(self, graph: DGLGraph, feat: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward computation."""
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting `allow_zero_in_degree` '
                               'to `True` when constructing this module will '
                               'suppress this check and let the users handle '
                               'it by themselves.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if hasattr(self, 'fc_src'):
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'ft': feat_src, 'el': (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)})
            graph.dstdata.update({'er': (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)

            # Apply normalization
            rst = rst.view(rst.shape[0], -1)
            rst = self.norm(rst)
            rst = rst.view(rst.shape[0], self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst

class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4, feat_drop=0.0, attn_drop=0.0, do_train=False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GAT, self).__init__()
        self.do_train = do_train
        self.device = torch.device(device)  # Convert string to PyTorch device

        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(GATConv(in_feats, out_feats // num_heads, num_heads, 
                                   feat_drop=feat_drop, attn_drop=attn_drop, 
                                   residual=True, activation=F.leaky_relu, 
                                   allow_zero_in_degree=True))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_feats * num_heads, out_feats // num_heads, num_heads, 
                                       feat_drop=feat_drop, attn_drop=attn_drop, 
                                       residual=True, activation=F.leaky_relu, 
                                       allow_zero_in_degree=True))
        
        self.predict = nn.Linear(out_feats * num_heads, 1)  # Adjust for multi-head concat
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=feat_drop)  # Use dropout probability from init

        self.to(self.device)  # Move model to device after initializing all layers

    def forward(self, g, features):
        g = g.to(self.device)
        features = features.to(self.device)  # Move features to same device
        h = features

        for layer in self.layers:
            # Print the device of g before moving it to the device
            ##print(f"g device before moving to {self.device}: {g.device}")
            
            g = g.to(self.device)  # Ensure g is on the correct device
            
            # Print the device of h before the layer
            ##print(f"h device before layer: {h.device}")
            
            h = self.dropout(h)
            h = layer(g, h).flatten(1)  # Flatten after GATConv
            h = self.leaky_relu(h)  # Apply LeakyReLU activation
            
            # Optionally, print the device of h after the operation
            ##print(f"h device after layer: {h.device}")


        if not self.do_train:
            return h.detach()  # Detach for inference

        logits = self.predict(h)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  

class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device  # Store the device information
        self.W1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.W2 = nn.Linear(hidden_size, 1).to(self.device)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1).to(self.device)  # Ensure `h` is on the correct device
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        g = g.to(self.device)  # Move the graph to the correct device
        h = h.to(self.device)  # Ensure `h` is moved to the correct device
        with g.local_scope():
            g.ndata['h'] = h  # Now assign `h` to the graph on the correct device
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, feat_drop=0.0, do_train=False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GCN, self).__init__()
        self.device = device
        self.do_train = do_train
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=feat_drop)

        # First GCN layer
        self.layers.append(GraphConv(in_feats, out_feats, activation=F.relu, allow_zero_in_degree=True))

        # Hidden GCN layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(out_feats, out_feats, activation=F.relu, allow_zero_in_degree=True))

        # Prediction layer
        self.predict = nn.Linear(out_feats, 1).to(device)

    def forward(self, g, features):
        g = g.to(self.device)  # Ensure graph is on the correct device
        features = features.to(self.device)  # Ensure features are on the correct device

        g = dgl.add_self_loop(g)  # Add self-loops to avoid zero in-degree issues
        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h)

        if not self.do_train:
            return h.detach()

        logits = self.predict(h)
        return logits


class GIN(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, feat_drop=0.0, do_train=False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GIN, self).__init__()
        self.device = device
        self.do_train = do_train
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=feat_drop)

        # MLP for GINConv
        def mlp():
            return nn.Sequential(
                nn.Linear(in_feats, out_feats),
                nn.ReLU(),
                nn.Linear(out_feats, out_feats),
                nn.ReLU()
            )

        # First GIN layer
        self.layers.append(GINConv(mlp(), 'sum'))

        # Hidden GIN layers
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(mlp(), 'sum'))

        # Prediction layer
        self.predict = nn.Linear(out_feats, 1).to(device)

    def forward(self, g, features):
        g = g.to(self.device)  # Ensure graph is on the correct device
        features = features.to(self.device)  # Ensure features are on the correct device

        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h)

        if not self.do_train:
            return h.detach()

        logits = self.predict(h)
        return logits


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(GraphSAGE, self).__init__()
        self.device = device
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean').to(device)
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean').to(device)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        ).to(device)

    def forward(self, g, features):
        g = g.to(self.device)  # Ensure graph is on the correct device
        features = features.to(self.device)  # Ensure features are on the correct device

        x = F.relu(self.sage1(g, features))
        x = F.relu(self.sage2(g, x))
        return self.mlp(x)


class ChebNet(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, k=3, feat_drop=0.0, do_train=False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(ChebNet, self).__init__()
        self.device = device
        self.do_train = do_train
        self.k = k  # Order of Chebyshev polynomials

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(ChebConv(in_feats, out_feats, k=self.k, activation=F.relu).to(device))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(ChebConv(out_feats, out_feats, k=self.k, activation=F.relu).to(device))

        # Prediction layer
        self.predict = nn.Linear(out_feats, 1).to(device)
        self.dropout = nn.Dropout(p=feat_drop)

    def forward(self, g, features):
        g = g.to(self.device)  # Ensure graph is on the correct device
        features = features.to(self.device)  # Ensure features are on the correct device

        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h)

        if not self.do_train:
            return h.detach()

        logits = self.predict(h)
        return logits
