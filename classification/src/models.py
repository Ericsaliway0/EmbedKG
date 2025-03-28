import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from dgl.nn import SAGEConv, GATConv, GraphConv, GINConv, ChebConv
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops

class EGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.3, epsilon=1e-4):
        """
        Efficient Graph Convolutional Network (EGCN) with:
        - Chebyshev Polynomial Approximation (Adaptive)
        - Early Stopping for Chebyshev Expansion
        - Three Aggregation Terms for Better Expressivity

        Parameters:
        - in_feats: Input feature size
        - hidden_feats: Hidden layer feature size
        - out_feats: Output feature size
        - k: Maximum Chebyshev polynomial order
        - dropout: Dropout rate for regularization
        - epsilon: Early stopping tolerance for Chebyshev computation
        """
        super(EGCN, self).__init__()
        self.k = k  # Maximum Chebyshev order
        self.dropout = dropout
        self.epsilon = epsilon  # Stopping threshold

        # Chebyshev Convolution Layers
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.cheb3 = ChebConv(hidden_feats, hidden_feats, k)  # Third aggregation term

        # Fully Connected Layer (MLP)
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
        Forward pass for EGCN with adaptive Chebyshev polynomial expansion.

        Parameters:
        - g: DGL graph
        - features: Input node features

        Returns:
        - Output predictions after passing through EGCN layers
        """
        # First Chebyshev Convolution
        x = F.relu(self.cheb1(g, features))
        x = self.norm(x)

        # Early stopping check for Chebyshev polynomials
        prev_x = x.clone()
        for _ in range(1, self.k):
            x_new = F.relu(self.cheb2(g, x))
            if torch.norm(x_new - prev_x) < self.epsilon:
                break  # Stop if change is small
            prev_x = x_new.clone()
        
        # Third Aggregation Term for Feature Enhancement
        x_res = x  # Residual Connection
        x = F.relu(self.cheb3(g, x))
        x = self.dropout_layer(x) + x_res  # Efficient residual connection

        return self.mlp(x)

class HGDC(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        self.linear1 = Linear(in_channels, hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2 * hidden_channels, 1)
        self.linear_r2 = Linear(2 * hidden_channels, 1)
        self.linear_r3 = Linear(2 * hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_edge(edge_index_1, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)
        edge_index_2, _ = dropout_edge(edge_index_2, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out

class MTGCN(torch.nn.Module):
    def __init__(self, args):
        super(MTGCN, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(58, 100)
        self.lin2 = Linear(58, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.5,
                                     force_undirected=True,
                                     num_nodes=data.x.size()[0],
                                     training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2

class EMOGI(torch.nn.Module):
    def __init__(self,args):
        super(EMOGI, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2)
        self.conv2 = ChebConv(300, 100, K=2)
        self.conv3 = ChebConv(100, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

class ChebNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3):
        """
        ChebNet implementation using DGL's ChebConv.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - k: Chebyshev polynomial order.
        """
        super(ChebNet, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for ChebNet.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through ChebNet layers.
        """
        x = F.relu(self.cheb1(g, features))
        x = F.relu(self.cheb2(g, x))
        return self.mlp(x)

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GIN, self).__init__()
        # Define the first GIN layer
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'  # Aggregation method: 'mean', 'max', or 'sum'
        )
        # Define the second GIN layer
        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'
        )
        # MLP for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        # Apply the first GIN layer
        x = F.relu(self.gin1(g, features))
        # Apply the second GIN layer
        x = F.relu(self.gin2(g, x))
        # Apply the MLP
        return self.mlp(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.sage1(g, features))
        x = F.relu(self.sage2(g, x))
        return self.mlp(x)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT).
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for GAT.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through GAT layers.
        """
        x = self.gat1(g, features)
        x = x.flatten(1)  # Flatten the output of multi-head attention
        x = self.gat2(g, x)
        x = x.flatten(1)  # Flatten the output again
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_feats, hidden_feats)
        self.gcn2 = GraphConv(hidden_feats, hidden_feats)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.gcn1(g, features))
        x = F.relu(self.gcn2(g, x))
        return self.mlp(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ##bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Ensure targets are of type float
        targets = targets.float()

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    