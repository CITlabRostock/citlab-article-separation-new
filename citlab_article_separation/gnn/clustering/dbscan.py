import numpy as np
import argparse
import json
import re


class DBScanRelation:
    """ DBSCAN based on Chris McCormicks https://github.com/chrisjmccormick/dbscan"""
    def __init__(self,
                 min_neighbors_for_cluster=1,
                 confidence_threshold=0.5,
                 cluster_agreement_threshold=0.5,
                 weight_handling='avg',
                 assign_noise_clusters=True):
        """
        DBScan algorithm on a confidence graph representing article relations.

        It will return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1.
        """

        assert weight_handling in ('avg', 'min', 'max'), \
            f"Weight handling {weight_handling} is not supported! Choose from ('avg', 'min', 'max') instead!"
        self.num_nodes = None
        self.confidences = None
        self.labels = None
        self.min_neighbors_for_cluster = min_neighbors_for_cluster
        self.confidence_threshold = confidence_threshold
        self.cluster_agreement_threshold = cluster_agreement_threshold
        self.weight_handling = weight_handling
        self.assign_noise_clusters = assign_noise_clusters

    def initialize_clustering(self, num_nodes, confidences):
        self.num_nodes = num_nodes
        assert self.num_nodes * self.num_nodes == confidences.flatten().shape[0], \
            f"Expected {self.num_nodes * self.num_nodes} relations for a graph with {self.num_nodes} nodes. " \
            f"Got {confidences.shape[0]} instead."
        self.confidences = np.reshape(np.copy(confidences), [self.num_nodes, self.num_nodes])

        # make sure confidences are symmetric
        if not np.array_equal(self.confidences, np.transpose(self.confidences)):
            cc = np.copy(self.confidences)
            if self.weight_handling == 'avg':
                self.confidences = (self.confidences + np.transpose(self.confidences)) / 2
            elif self.weight_handling == 'max':
                self.confidences = np.stack([self.confidences, np.transpose(self.confidences)], axis=-1)
                self.confidences = np.max(self.confidences, axis=-1)
            elif self.weight_handling == 'min':
                self.confidences = np.stack([self.confidences, np.transpose(self.confidences)], axis=-1)
                self.confidences = np.min(self.confidences, axis=-1)
            print(f"Confidence matrix is forced to be symmetric, by taking '{self.weight_handling}' of pairs.")
            print("Average symmetry deviation: ", np.mean(np.abs(np.around(cc - self.confidences, decimals=4))))
            print("Maximum symmetry deviation: ", np.max(np.abs(np.around(cc - self.confidences, decimals=4))))

        # This list holds the cluster assignment for each node in the graph
        #    -1 -> Indicates a noise node
        #     0 -> Means the node hasn't been considered yet
        #    1+ -> Clusters are numbered starting from 1
        # Initially all labels are 0.
        self.labels = [0] * self.num_nodes

    def cluster_relations(self, num_nodes, confidences):
        """Clusters the graph nodes with DBScan according to the given relation confidences."""
        # Initialization
        self.initialize_clustering(num_nodes, confidences)

        # Current cluster
        label = 0
        # Look for valid center points and create a new cluster
        for node_index in range(self.num_nodes):
            # Only nodes that haven't been considered yet can be picked as a new center point
            if not (self.labels[node_index] == 0):
                continue

            # Find all neighboring nodes
            neighbor_nodes = self.region_query(node_index)

            # If the number is below 'min_nodes_for_cluster', this node is "noise"
            # A "noise" node may later be picked up by another cluster as a boundary node
            if len(neighbor_nodes) < self.min_neighbors_for_cluster:
                self.labels[node_index] = -1

            # Otherwise, this node is a center node for a new cluster
            else:
                label += 1
                # Build the cluster
                self.grow_cluster(node_index, neighbor_nodes, label)

        # Every "noise" node gets its own cluster
        if self.assign_noise_clusters:
            self.create_clusters_for_noise_nodes(label)

        # All data has been clustered
        return self.labels

    def grow_cluster(self, node_index, neighbor_nodes, label):
        """Grow a new cluster with label 'label' from a center node with index 'node_index'."""

        # Assign the cluster label to the center node
        self.labels[node_index] = label

        # Look at each neighbor of the center node. Neighbor nodes will be used as a FIFO queue
        # (while-loop) of nodes to search, which grows as we discover new nodes for the cluster
        i = 0
        while i < len(neighbor_nodes):
            # Next node from the queue
            neighbor = neighbor_nodes[i]

            # If 'neighbor' was labelled "noise" then it's not a center node (not enough neighbors),
            # so make it a boundary node of the cluster and move on
            if self.labels[neighbor] == -1:
                if self.validate_cluster_agreement(neighbor, label):
                    self.labels[neighbor] = label

            # Otherwise, if 'neighbor' isn't already claimed, add the node to the cluster
            elif self.labels[neighbor] == 0:
                if self.validate_cluster_agreement(neighbor, label):
                    self.labels[neighbor] = label

                    # Find all the neighbors of 'neighbor'
                    next_neighbor_nodes = self.region_query(neighbor)

                    # If 'neighbor' has at least 'min_nodes_for_cluster' neighbors, it's a new center node.
                    # Add all of its neighbors to the FIFO queue to be searched
                    if len(next_neighbor_nodes) >= self.min_neighbors_for_cluster:
                        neighbor_nodes += next_neighbor_nodes

            # Next node in the FIFO queue
            i += 1

    def region_query(self, node_index):
        """
        Find all nodes within the defined neighborhood of the node 'node_index'.
        The Neighborhood is given by a certain confidence threshold.
        """
        neighbor_confidences = self.confidences[node_index, :]
        neighbor_indices = neighbor_confidences > self.confidence_threshold
        neighbors = np.arange(self.num_nodes)[np.where(neighbor_indices)]
        neighbors = neighbors.tolist()
        if node_index in neighbors:  # remove self-loop as neighbor
            neighbors.remove(node_index)
        return neighbors

    def validate_cluster_agreement(self, node, label):
        cluster_indices = [l == label for l in self.labels]  # node indices with matching label
        cluster_confidences = self.confidences[node, cluster_indices]  # confidences between node and cluster_nodes
        cluster_agreement = np.mean(cluster_confidences)  # average confidence
        return cluster_agreement > self.cluster_agreement_threshold  # confidence greater than threshold?

    def create_clusters_for_noise_nodes(self, label):
        for index in range(len(self.labels)):
            if self.labels[index] == -1:
                label += 1
                self.labels[index] = label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to json file containing graph confidences")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Threshold in [0,1] that determines node neighborhoods via confidence cutoff")
    parser.add_argument("--cluster_agreement_threshold", type=float, default=0.5,
                        help="Threshold in [0,1] that determines if a new node is assigned to an existing cluster. "
                             "The average confidence score to all nodes of the cluster is used.")
    parser.add_argument("--min_neighbors_for_cluster", type=int, default=0,
                        help="Minimum number of neighbors a node has to have to build a new cluster.")
    parser.add_argument("--weight_handling", type=str, default="avg",
                        help="Handles the conversion of the DiGraph confidences to UniGraph confidences. "
                             "Available options are ('avg', 'min', 'max')")
    args = parser.parse_args()

    assert args.weight_handling in ('avg', 'min', 'max'), \
        f"Unknown weight handling. Choose from ('avg', 'min', 'max') instead!"

    json_path = args.json_path
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
        print(f"Loading article_ids from json file {json_path}")

    dbscan = DBScanRelation(min_neighbors_for_cluster=args.min_neighbors_for_cluster,
                            confidence_threshold=args.confidence_threshold,
                            cluster_agreement_threshold=args.cluster_agreement_threshold,
                            weight_handling=args.weight_handling)

    for page_name in json_data:
        # print("\n" + page_name)
        page_data = json_data[page_name]
        num_nodes = len(page_data.keys())
        # print("num_nodes", num_nodes)
        confidences = np.empty((num_nodes, num_nodes))
        for i, text_region in enumerate(page_data):
            text_region_confidences = page_data[text_region]['confidences']
            for j, tr in enumerate(text_region_confidences):
                confidences[i, j] = text_region_confidences[tr]

        clustering = dbscan.cluster_relations(num_nodes, confidences)
        # print(clustering)
        # print("len_clustering", len(clustering))

        # save new article_ids from clustering
        for i, text_region in enumerate(page_data):
            json_data[page_name][text_region]['article_id'] = "a" + str(clustering[i])
            del json_data[page_name][text_region]['confidences']

    save_path = re.sub(r'confidences', f'DBScanMod_t{args.confidence_threshold}a{args.cluster_agreement_threshold}{args.weight_handling}', json_path)
    with open(save_path, "w") as out_file:
        json.dump(json_data, out_file, indent=4)
        print(f"\nWrote json dump: {save_path}.")

