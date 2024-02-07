from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core.swc import read_swc
from ssl_neuron.data.data_utils import connect_graph, remove_axon, rotate_cell
from ssl_neuron.utils import neighbors_to_adjacency, plot_neuron, remap_neighbors


if __name__ == "__main__":
    cell_ids = list(np.load("all_ids.npy"))
    ctc = CellTypesCache(manifest_file="cell_types/manifest.json")
    dpath = list(Path.cwd().parents)[2] / "data/allen_cell_types"
    metadata_file = f"{dpath}/info/41593_2019_417_MOESM5_ESM.xlsx"
    df = pd.read_excel(metadata_file)
    allen_meta = pd.read_csv(f"{dpath}/info/ACT_info_swc.csv")
    specimen_to_swc = dict(zip(allen_meta["specimen__id"], allen_meta["swc__fname"]))
    
    for cell_id in tqdm(cell_ids):
        path = Path("./skeletons/", str(cell_id))
        path.mkdir(parents=True, exist_ok=True)
        morphology = read_swc(f"{dpath}/raw/{specimen_to_swc[cell_id]}")
        # Rotate respecitve to pia.
        morphology = rotate_cell(cell_id, morphology, df)
        # Get soma coordinates.
        soma = morphology.soma
        soma_pos = np.array([soma["x"], soma["y"], soma["z"]])
        soma_id = soma["id"]
        # Process graph.
        neighbors = {}
        idx2node = {}
        for i, item in enumerate(morphology.compartment_list):
            # Get node features.
            sec_type = [0, 0, 0, 0]
            sec_type[item["type"] - 1] = 1
            feat = tuple([item["x"], item["y"], item["z"], item["radius"]]) + tuple(sec_type)
            idx2node[i] = feat
            # Get neighbors.
            neighbors[i] = set(item["children"])
            if item["parent"] != -1:
                neighbors[i].add(item["parent"])
        features = np.array(list(idx2node.values()))
        assert ~np.any(np.isnan(features))
        # Normalize soma position to origin.
        norm_features = features.copy()
        norm_features[:, :3] = norm_features[:, :3] - soma_pos
        
        # Test if graph is connected.
        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))
        G = nx.Graph(adj_matrix)
        if nx.number_connected_components(G) > 1:
            adj_matrix, neighbors = connect_graph(adj_matrix, neighbors, features)

        assert len(neighbors) == len(adj_matrix)
        # Remove axons.
        neighbors, norm_features, soma_id = remove_axon(neighbors, norm_features, int(soma_id))
        assert len(neighbors) == len(norm_features)
        assert ~np.any(np.isnan(norm_features))

        np.save(Path(path, "features"), norm_features)
        with open(Path(path, "neighbors.pkl"), "wb") as f:
            pickle.dump(dict(neighbors), f, pickle.HIGHEST_PROTOCOL)
    