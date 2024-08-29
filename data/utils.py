import os
import h5py
import uuid
import torch
import random
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import Data
from simg.model_utils import pipeline
from simg.data import get_connectivity_info
from simg.graph_construction import convert_NBO_graph_to_downstream
from functools import partial

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

def convert2graph(xyz):
    smi, mol = xyz
    try:
        xyz_data = [l + '\n' for l in mol.split('\n')[2:-1]]
        symbols = [l.split()[0] for l in xyz_data]
        coordinates = np.array([[float(num) for num in l.strip().split()[1:]] for l in xyz_data])
        connectivity = get_connectivity_info(xyz_data)
        graph, _, _, _ = pipeline(symbols, coordinates, connectivity)
        graph.smiles = smi

        return graph
    except Exception as e:
        # skip graphs with errors
        print("Error while processing SMILES: ", smi)
        print(e)


def smiles_to_xyz(smiles):
    xyzs = Parallel(n_jobs=32)(delayed(convert2xyz)(smi) for smi in tqdm(smiles))

    return xyzs

def xyz_to_graph(xyzs):
    graphs = Parallel(n_jobs=32)(delayed(convert2graph)(xyz) for xyz in tqdm(xyzs))
    graphs = [graph for graph in graphs if graph is not None] # remove None types

    return graphs

def graph_to_NBO(graphs, molecular_only):
    nbo_graphs = Parallel(n_jobs=32)(delayed(partial(convert_NBO_graph_to_downstream, molecular_only=molecular_only))(graph) for graph in tqdm(graphs))

    return nbo_graphs

def preprocess(args):
    print("Preparing dataset...")
    data = prepare_savoie(args.size)
    smiles = list(data.keys())
    
    print("Converting SMILES to XYZ...")
    xyzs = smiles_to_xyz(smiles)

    print("Converting XYZ to graphs...")
    graphs = xyz_to_graph(xyzs)

    if args.from_NBO:
        print('Converting to downstream format...')
        graphs = graph_to_NBO(graphs, args.molecular)
        
    print('Cleaning up and combining spectrum...')
    clean_graphs = []
    for graph in tqdm(graphs):
        clean_graphs.append(
            Data(
                x=torch.FloatTensor(graph.x),
                edge_index=torch.LongTensor(graph.edge_index),
                edge_attr=torch.FloatTensor(graph.edge_attr),
                smiles=graph.smiles,
                y=data[graph.smiles]
            )
        )

    return clean_graphs