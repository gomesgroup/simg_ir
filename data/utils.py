import os
import h5py
import uuid
import random
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import Data
from simg.model_utils import pipeline
from simg.data import get_connectivity_info

def prepare_savoie(size, method="ir", filename="data/savoie_dataset.hdf5"):
    data = {}
    with h5py.File(filename, "r") as f:
        # available methods within the dataset: 'ir', 'ms', 'nmr'
        group = f[method]
        smiles = list(group.keys())

        # get random subset of smiles, else get everything
        if size is not None:
            smiles_samples = random.sample(smiles, size)
        else:
            smiles_samples = smiles
        
        # combine spectrum data
        for smi in smiles_samples:
            data[smi] = group[smi][()]
        
    return data

def convert2xyz(smi):
    path = uuid.uuid4().hex
    xyz_path = f"data/{path}.xyz"
    smi_path = f"data/{path}.smi"

    with open(smi_path, "w") as f:
        f.write(smi + '\n')

    os.system(f'obabel -i smi {smi_path} -o xyz -O {xyz_path} --gen3d >/dev/null 2>&1')

    with open(xyz_path, "r") as f:
        xyz = f.read()

    os.remove(xyz_path)
    os.remove(smi_path)

    return (smi, xyz)

def smiles_to_xyz(smiles):
    xyzs = Parallel(n_jobs=128)(delayed(convert2xyz)(smi) for smi in tqdm(smiles))

    return xyzs

def xyz_to_graph(xyzs):
    for i in range(0, len(xyzs), 10_000):
        mols = xyzs[i: i + 10_000]
        graphs = []

        for smi, mol in tqdm(mols):
            xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
            symbols = [l.split()[0] for l in xyz_data]
            coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
            connectivity = get_connectivity_info(xyz_data)
            nbo_graph, _, _, _ = pipeline(symbols, coordinates, connectivity)

            # only get the necessary attributes to save on disk space
            graph = Data(x=nbo_graph.x, 
                         edge_index=nbo_graph.edge_index,
                         edge_attr=nbo_graph.edge_attr,
                         y=nbo_graph.y,
                         smiles=smi)
            graphs.append(graph)

    return graphs

def preprocess(size):
    data = prepare_savoie(size)
    smiles = list(data.keys())
    xyzs = smiles_to_xyz(smiles)
    graphs = xyz_to_graph(xyzs)

    # combine spectrum data
    for graph in graphs:
        graph.y = data[graph.smiles]

    return graphs