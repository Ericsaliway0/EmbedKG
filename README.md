# PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions
This repository provides the code for our research project "PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions".


## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) https://www.biosino.org/dbDEMC/index


## Scripts


## Setup
-)conda create -n kg python=3.10 -y

-)conda activate kg

-)pip install -r requirements.txt


## Get start
## get embedding
python PERGAT_embedding/gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 105

<<<<<<< HEAD
## prediction
python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

=======

python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

>>>>>>> 007709138d8c23aac23bc2af32000b59e982b983
