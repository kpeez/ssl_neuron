import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ssl_neuron.utils import (
    compute_node_distances,
    drop_random_branch,
    get_leaf_branch_nodes,
    jitter_node_pos,
    neighbors_to_adjacency_torch,
    remap_neighbors,
    rotate_graph,
    subsample_graph,
    translate_soma_pos,
)


class GraphDataset(Dataset):
    """Dataset of neuronal graphs.

    Neuronal graphs are assumed to be soma-centered (i.e. soma node
    position is (0, 0, 0) and axons have been removed. Node positions
    are assumed to be in microns and y-axis is orthogonal to the pia.
    """

    def __init__(
        self,
        config: Mapping[str, Mapping[str, str]],
        mode: str = "train",
        inference: bool = False,
    ) -> None:
        self.config = config
        self.mode = mode
        self.inference = inference
        data_path = config["data"]["path"]

        # Augmentation parameters.
        self.jitter_var = config["data"]["jitter_var"]
        self.rotation_axis = config["data"]["rotation_axis"]
        self.n_drop_branch = config["data"]["n_drop_branch"]
        self.translate_var = config["data"]["translate_var"]
        self.n_nodes = config["data"]["n_nodes"]

        # Load cell ids.
        cell_ids = list(np.load(Path(f"{data_path}/{mode}_ids.npy").resolve()))

        # Load graphs.
        self.manager = Manager()
        self.cells = self.manager.dict()
        count = 0
        for cell_id in tqdm(cell_ids):
            # Adapt for datasets where this is not true.
            soma_id = 0

            features = np.load(Path(data_path, "skeletons", str(cell_id), "features.npy"))
            with open(Path(data_path, "skeletons", str(cell_id), "neighbors.pkl"), "rb") as f:
                neighbors = pickle.load(f)

            assert len(features) == len(neighbors)

            if len(features) >= self.n_nodes or self.inference:
                # Subsample graphs for faster processing during training.
                neighbors, not_deleted = subsample_graph(
                    neighbors=neighbors,
                    not_deleted=set(range(len(neighbors))),
                    keep_nodes=1000,
                    protected=[soma_id],
                )
                # Remap neighbor indices to 0..999.
                neighbors, subsampled2new = remap_neighbors(neighbors)
                soma_id = subsampled2new[soma_id]

                # Accumulate features of subsampled nodes.
                features = features[list(subsampled2new.keys()), :3]

                leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
                # Using the distances we can infer the direction of an edge.
                distances = compute_node_distances(soma_id, neighbors)

                item = {
                    "cell_id": cell_id,
                    "features": features,
                    "neighbors": neighbors,
                    "distances": distances,
                    "soma_id": soma_id,
                    "leaf_branch_nodes": leaf_branch_nodes,
                }

                self.cells[count] = item
                count += 1

        self.num_samples = len(self.cells)

    def __len__(self) -> int:
        return self.num_samples

    def _delete_subbranch(
        self,
        neighbors: Mapping[str, list],
        distances: Mapping[str, list],
        leaf_branch_nodes: list[int],
    ) -> set[int]:
        """Delete random subbranches from the graph.

        Args:
            neighbors (dict[str, list]): dict of neighbors per node
            distances (dict[str, float]): dict of distances of nodes to origin
            leaf_branch_nodes (_type_): _description_

        Returns:
            tuple[]: _description_
        """
        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors)))
        for _ in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(
                leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes
            )
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes

            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        # Delete random branches.
        not_deleted = self._delete_subbranch(neighbors2, distances, leaf_branch_nodes)

        # Subsample graphs to fixed number of nodes.
        neighbors2, not_deleted = subsample_graph(
            neighbors=neighbors2,
            not_deleted=not_deleted,
            keep_nodes=self.n_nodes,
            protected=(soma_id),
        )

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)

        assert adj_matrix.shape == (
            self.n_nodes,
            self.n_nodes,
        ), f"Adjacency matrix has shape: {adj_matrix.shape}"

        return neighbors2, adj_matrix, not_deleted

    def _augment_node_position(self, features):
        # Extract positional features (xyz-position).
        pos = features[:, :3]

        # Rotate (random 3D rotation or rotation around specific axis).
        rot_pos = rotate_graph(pos, axis=self.rotation_axis)

        # Randomly jitter node position.
        jittered_pos = jitter_node_pos(rot_pos, scale=self.jitter_var)

        # Translate neuron position as a whole.
        jittered_pos = translate_soma_pos(jittered_pos, scale=self.translate_var)

        features[:, :3] = jittered_pos

        return features

    def _augment(self, cell):
        features = cell["features"]
        neighbors = cell["neighbors"]
        distances = cell["distances"]

        # Reduce nodes to N == n_nodes via subgraph deletion and subsampling.
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(
            neighbors, [int(cell["soma_id"])], distances, cell["leaf_branch_nodes"]
        )

        # Extract features of remaining nodes.
        new_features = features[not_deleted].copy()

        # Augment node position via rotation and jittering.
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix

    def __getsingleitem__(self, index):
        cell = self.cells[index]
        return cell["features"], cell["neighbors"]

    def __getitem__(self, index):
        cell = self.cells[index]

        # Compute two different views through augmentations.
        features1, adj_matrix1 = self._augment(cell)
        features2, adj_matrix2 = self._augment(cell)

        return features1, features2, adj_matrix1, adj_matrix2


def build_dataloader(config, use_cuda: bool = False):
    if torch.cuda.is_available():
        use_cuda = True

    kwargs = (
        {
            "num_workers": config["data"]["num_workers"],
            "pin_memory": True,
            "persistent_workers": False,
        }
        if use_cuda
        else {}
    )

    train_loader = DataLoader(
        GraphDataset(config, mode="train"),
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    val_dataset = GraphDataset(config, mode="val")
    batch_size = (
        val_dataset.num_samples
        if val_dataset.__len__() < config["data"]["batch_size"]
        else config["data"]["batch_size"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )

    return train_loader, val_loader
