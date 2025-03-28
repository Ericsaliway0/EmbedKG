import argparse
import torch
##from src.data_loader import load_graph_data
from src.train import train, plot_and_analyze

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="GNN-based driver gene prediction")
    parser.add_argument('--in_feats', type=int, default=2048)
    parser.add_argument('--hidden_feats', type=int, default=1024)
    parser.add_argument('--out_feats', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_type', type=str, choices=['GraphSAGE', 'GAT', 'GCN', 'GIN', 'ChebNet', 'HGDC', 'EMOGI', 'MTGCN', 'EGCN'], required=True)
    parser.add_argument('--net_type', type=str, choices=['CPDB', 'STRING', 'HIPPIE'], required=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--score_threshold', type=float, default=0.99)
    args = parser.parse_args()
    
    train(args)
    plot_and_analyze(args)


## EGCN % python main.py --model_type EGCN --net_type CPDB --score_threshold 0.99 --in_feats 256 --hidden_feats 128 --learning_rate 0.001 --num_epochs 5005
## (kg39) ericsali@erics-MacBook-Pro-4 EGCN % python main.py --model_type EGCN --net_type CPDB --score_threshold 0.99 --in_feats 2048 --hidden_feats 1024 --learning_rate 0.001 --num_epochs 500