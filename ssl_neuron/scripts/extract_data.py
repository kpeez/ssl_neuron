import pickle
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
from allensdk.core.swc import read_swc
from tqdm import tqdm

from ssl_neuron.data.data_utils import connect_graph, remove_axon
from ssl_neuron.utils import neighbors_to_adjacency


def read_morphology(swc_file: str | Path) -> tuple[dict, np.ndarray]:
    """Reads the morphology data from an SWC file for a given cell ID."""
    morphology = read_swc(swc_file)
    return morphology


def process_morphology(morphology: Sequence[Mapping, np.ndarray]) -> tuple[dict, np.ndarray]:
    """Processes the morphology to extract and normalize features."""
    soma = morphology.soma
    soma_pos = np.array([soma["x"], soma["y"], soma["z"]])
    soma_id = soma["id"]
    assert soma_id is not None
    neighbors = {}
    idx2node = {}
    for i, item in enumerate(morphology.compartment_list):
        sec_type = [0, 0, 0, 0]
        sec_type[item["type"] - 1] = 1
        feat = (item["x"], item["y"], item["z"], item["radius"], *sec_type)
        idx2node[i] = feat
        neighbors[i] = set(item["children"])
        if item["parent"] != -1:
            neighbors[i].add(item["parent"])

    features = np.array(list(idx2node.values()))
    assert not np.any(np.isnan(features))
    norm_features = features.copy()
    norm_features[:, :3] -= soma_pos
    # Ensure the graph is connected
    adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))
    G = nx.Graph(adj_matrix)
    if nx.number_connected_components(G) > 1:
        adj_matrix, neighbors = connect_graph(adj_matrix, neighbors, features)
    neighbors, norm_features, _ = remove_axon(neighbors, norm_features, int(soma_id))
    assert len(neighbors) == len(norm_features) and not np.any(np.isnan(norm_features))

    return neighbors, norm_features


def save_processed_data(dpath: Path, neighbors: Mapping, norm_features: np.ndarray) -> None:
    """Saves the processed data to disk."""
    dpath.mkdir(parents=True, exist_ok=True)
    np.save(Path(dpath, "features.npy"), norm_features)
    with open(Path(dpath, "neighbors.pkl"), "wb") as f:
        pickle.dump(neighbors, f, pickle.HIGHEST_PROTOCOL)


def process_cell(swc_file: str | Path, output_dir: str | Path | None = None) -> bool:
    """Processes a single cell, orchestrating the reading, processing, and saving of data.

    Args:
        swc_file (str): Path to the SWC file.
        output_dir (str, optional): Output directory. Defaults to parent of `swc_file` if None.
    """
    swc_file = Path(swc_file)
    cell_id = str(swc_file.stem)
    export_dir = output_dir or swc_file.parent
    dpath = Path(f"{export_dir}/skeletons/{cell_id}")
    try:
        morphology = read_morphology(swc_file)
        neighbors, norm_features = process_morphology(morphology)
        save_processed_data(dpath, neighbors, norm_features)
        return True
    except Exception as e:
        print(f"Failed to process: {swc_file.stem} error: {e}")
        return False


def process_cells(
    swc_files: Sequence[str | Path],
    split: str = "all",
    output_dir: str | Path | None = None,
) -> None:
    """Processes all cells in a given directory.

    Args:
        swc_files (Sequence[str | Path]): Path to the SWC files.
        split (str, optional): Split to process. Defaults to "all".
        output_dir (str | Path, optional): Output directory. Defaults to parent of `swc_files` if None.
    """
    output_dir = Path(output_dir) if output_dir else Path(swc_files[0]).parent
    processed = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_cell, file, output_dir): Path(file) for file in swc_files
        }
        for future in tqdm(as_completed(futures), total=len(swc_files)):
            file = futures[future]
            if future.result():
                processed.append(file.stem)

    print(f"Processed {len(processed)} cells. Saving IDs to {output_dir}/{split.lower()}_ids.npy")
    np.save(f"{output_dir}/{split}_ids.npy", processed)


if __name__ == "__main__":
    from typer import Argument, Option, Typer

    app = Typer()

    @app.command()
    def extract_cell_data(
        swc_path: str = Argument(..., help="Path to the SWC files."),
        split: str = Option("all", "-s", "--split", help="Split to process.", show_default=True),
        output_dir: str = Option(
            None, "-o", "--output-dir", help="Output directory.", show_default=True
        ),
    ) -> None:
        swc_files = list(Path(swc_path).glob("*.swc"))
        print("Processing", len(swc_files), "files...")
        process_cells(swc_files, split, output_dir)

    app()
