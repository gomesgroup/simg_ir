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
        if size != -1:
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
    xyzs = Parallel(n_jobs=32)(delayed(convert2xyz)(smi) for smi in tqdm(smiles))
    # xyzs = [convert2xyz(smi) for smi in tqdm(smiles)]

    return xyzs

def xyz_to_graph(xyzs):
    graphs = []
    for smi, mol in tqdm(xyzs):
        try:
            xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
            symbols = [l.split()[0] for l in xyz_data]
            coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
            connectivity = get_connectivity_info(xyz_data)
            graph, _, _, _ = pipeline(symbols, coordinates, connectivity)
            graph.smiles = smi
            graphs.append(graph)
        except:
            # skip graphs with errors
            print("Error while processing SMILES: ", smi)
            continue

    return graphs

def preprocess(size):
    print("Preparing dataset...")
    data = prepare_savoie(size)
    smiles = list(data.keys())
    
    print("Converting SMILES to XYZ...")
    xyzs = smiles_to_xyz(smiles)

    print("Converting XYZ to graphs...")
    graphs = xyz_to_graph(xyzs)

    # combine spectrum data
    print("Combining spectrum ")
    for graph in graphs:
        graph.y = data[graph.smiles]

    return graphs