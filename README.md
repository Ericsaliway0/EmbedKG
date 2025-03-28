## A Knowledge Graph-Based Graph Neural Network Framework for Multi-Omics Applications,

This repository contains the code for our project,  
**"A Knowledge Graph-Based Graph Neural Network Framework for Multi-Omics Applications."** 


![Alt text](images/__overview_framework.png)


## Data Source

The dataset is obtained from the following sources:

- **[miRTarBase](https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/index.php)**  
- **[TarBase](https://dianalab.e-ce.uth.gr/tarbasev9/downloads)**  
- **[miRNet](https://www.mirnet.ca/)**  

These databases provide curated and integrated protein-protein interaction (PPI) and pathway data for bioinformatics research.


## Setup and Get Started

1. Install the required dependencies:
   - `pip install -r requirements.txt`

2. Activate your Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install the necessary Python packages:
   - `pip install pandas`
   - `pip install py2neo pandas matplotlib scikit-learn`
   - `pip install tqdm`
   - `pip install seaborn`

5. Install DGL:
   - `conda install -c dglteam dgl`

6. Download the data from the built gene association graph using the link below and place it in the `data` directory before training:
   - [Download built graphs](https://drive.google.com/drive/folders/1AQWP7o5YgnHJLyrX08ORwGE85JDmCMo2?usp=drive_link)

7. To train the model for prediction, run the following command:
   - `python main.py  --model_type GAT --net_type miRTarBase --in-feats 256 --out-feats 256 --num-heads 2 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 201`

8. To train the model for classification, run the following command:
   - `python main.py --model_type EGCN --net_type CPDB --score_threshold 0.99 --in_feats 256 --hidden_feats 128 --learning_rate 0.001 --num_epochs 200`
