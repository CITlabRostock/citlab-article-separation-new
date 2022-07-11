# Example Networks

## Heading Detection
The `heading_detection_net.pb` contains an ARU-Net trained for heading detection.

## Seperator Detection
The `separator_detection_net.pb` contains an ARU-Net trained for separator detection.

# Graph Neural Network
The `mixed_gnn_vn7e2.pb` contains a Graph Neural Network for the relation prediction task, that is trained on a 
mix of German, Finnish and French newspapers. The name `vn7e2` describes the setup of the features that were used to 
train the GNN:
- `v`: Visual features were added during the training. This means that net network contains an ARU-Net branch, that 
is able to extract addtional visual feature information from given images corresponding to the newspaper pages. This 
is optional during the clustering process (see main README for further details).
- `n7e2`: The network was trained on 7 node features and 2 edge features. From the standard 15 node features (see 
main README), 8 were dismissed. In particular, the node features for the top and bottom baseline of each text block 
were masked out. A clustering process using this network, if all 15 features were generated, would look something 
like this
```bash
python -u run_gnn_clustering.py \
  --model_dir "path/to/mixed_gnn_vn7e2.pb" \
  --eval_list "path/to/json/list" \
  --input_params \
  node_feature_dim=15 \
  node_input_feature_mask=[1,1,1,1,0,0,0,0,0,0,0,0,1,1,1] \
  edge_feature_dim=2 \
  --clustering_method CLUSTERING_METHOD
```
The `mixed_gnn_vn7e3.pb` uses additional BERT language model features between text blocks as a third edge feature.

