import argparse
import os
from src.data_loader import load_graph_data
from src.train import train_and_evaluate
import torch

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='MLP Predictor')
    parser.add_argument('--in-feats', type=int, default=128, help='Dimension of the first layer')
    parser.add_argument('--hidden_feats', type=int, default=128)
    parser.add_argument('--out-feats', type=int, default=128, help='Dimension of the final layer')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
    parser.add_argument('--feat-drop', type=float, default=0.5, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='Attention dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for regularization')
    parser.add_argument('--chebyshev_order', type=int, default=3, help='Maximum Chebyshev polynomial order k')
    parser.add_argument('--model_type', type=str, choices=['GraphSAGE', 'GAT', 'GCN', 'GIN', 'ChebNet', 'EGCN'], required=True)
    parser.add_argument('--net_type', type=str, choices=['TarBase', 'miRTarBase', 'miRNet'], required=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_embeddings_hmdd_v3.json')
    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_embeddings_.json')
    ###G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_network.json')
    
    data_path = os.path.join('data/', f'{args.net_type}_miRNA_gene_embeddings.json')
    G_dgl, node_features, node_id_to_name = load_graph_data(data_path)
    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/_miRNA_gene_embeddings_miRTarBase.json')
    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/_miRNA_gene_embeddings_tarbase.json')
    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/_miRNA_disease_embeddings_dbDEMC_v3.json')
    ##G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_embeddings_HMDD.json')
    ##print('node_features.shape============\n',node_features)
    ##loocv_train_and_evaluate(args, G_dgl, node_features, node_id_to_name)
    train_and_evaluate(args, G_dgl, node_features, node_id_to_name)

 
## python main.py --in-feats 256 --out-feats 256 --num-heads 2 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 201 --model_type GAT --net_type miRTarBase   
## python main.py --in-feats 256 --out-feats 256 --num-heads 2 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 101
## PERGAT % python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 114    
'''## python link_prediction_main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 111     
## python link_prediction_main.py --in-feats 256 --out-feats 256 --num-heads 2 --num-layers 2 --lr 0.00005 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 1055
    
## python link_prediction_main.py --in-feats 128 --out-feats 128 --num-heads 2 --num-layers 2 --lr 0.00005 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 1055

## python link_prediction_main.py --in-feats 128 --out-feats 64 --num-heads 2 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 206

## HMDD python link_prediction_main.py --in-feats 128 --out-feats 256 --num-heads 2 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 1031

## python link_prediction_main.py --in-feats 128 --out-feats 256 --num-heads 4 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 1

## dbDEMC python link_prediction_main.py --in-feats 128 --out-feats 128 --num-heads 4 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 20'''