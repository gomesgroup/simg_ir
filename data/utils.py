import os
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
import json
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import h5py

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

def prepare_simulated(filename):
    with open(filename, 'r') as f:
        simulated_data = json.load(f)

    # Preprocess all spectra with interpolation and Gaussian smoothing
    new_wavenumbers = np.arange(400, 4000, 4)  # Common wavenumber grid
    processed_spectra = {}
    max_intensities = [np.max(data_obj['data']['intensity']) for data_obj in simulated_data]
    median_max_intensity = np.median(max_intensities)

    for data_obj in simulated_data:
        # Get original data
        orig_wavenumbers = np.array(data_obj['data']['wavenumber'])
        orig_intensities = np.array(data_obj['data']['intensity'])
        
        # Interpolate to new grid
        f = interpolate.interp1d(orig_wavenumbers, orig_intensities,
                                bounds_error=False, fill_value=0)
        new_intensities = f(new_wavenumbers)
        
        # Apply Gaussian smoothing with sigma=1
        smoothed = gaussian_filter1d(new_intensities, sigma=1)

        # remove negative values
        smoothed = np.where(smoothed < 0, 0, smoothed)  
        
        # Store processed spectrum
        processed_spectra[data_obj['smiles']] = smoothed

        # Scale down by median of max intensity
        processed_spectra[data_obj['smiles']] = processed_spectra[data_obj['smiles']] / median_max_intensity
    
    return processed_spectra

def prepare_difference(gas_filename, liquid_filename):
    gas_processed_spectra = prepare_simulated(gas_filename)
    liquid_processed_spectra = prepare_simulated(liquid_filename)
        
    # Create a dictionary to store the differences between liquid and gas spectra
    spectral_differences = {}

    # Calculate differences for all molecules
    common_smiles = set(gas_processed_spectra.keys()) & set(liquid_processed_spectra.keys())
    for smiles in common_smiles:
        gas_spectrum = gas_processed_spectra[smiles]
        liquid_spectrum = liquid_processed_spectra[smiles]

        # normalize by max intensity of gas spectrum
        gas_max = np.max(gas_spectrum)
        gas_spectrum = gas_spectrum / gas_max
        liquid_spectrum = liquid_spectrum / gas_max

        difference = liquid_spectrum - gas_spectrum
        spectral_differences[smiles] = difference

    return gas_processed_spectra, liquid_processed_spectra, spectral_differences

def prepare_correlation(gas_filename, liquid_filename):
    gas_processed_spectra = prepare_simulated(gas_filename)
    liquid_processed_spectra = prepare_simulated(liquid_filename)
        
    # Create a dictionary to store the differences between liquid and gas spectra
    spectral_correlations = {}

    # Calculate differences for all molecules
    common_smiles = set(gas_processed_spectra.keys()) & set(liquid_processed_spectra.keys())
    for smiles in common_smiles:
        gas_spectrum = gas_processed_spectra[smiles]
        liquid_spectrum = liquid_processed_spectra[smiles]
        correlation = np.corrcoef(gas_spectrum, liquid_spectrum)[0, 1]
        spectral_correlations[smiles] = correlation

    return gas_processed_spectra, liquid_processed_spectra, spectral_correlations

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
    gas_spectra = None
    liquid_spectra = None
    if args.dataset == "simulated":
        print("Preparing simulated dataset...")
        data = prepare_simulated(args.filename)
    elif args.dataset == "difference":
        print("Preparing difference dataset...")
        gas_spectra, liquid_spectra, data = prepare_difference(args.gas_filename, args.liquid_filename)
    elif args.dataset == "correlation":
        print("Preparing correlation dataset...")
        gas_spectra, liquid_spectra, data = prepare_correlation(args.gas_filename, args.liquid_filename)
    elif args.dataset == "savoie":
        print("Preparing Savoie dataset...")
        data = prepare_savoie(args.size, args.method, args.filename)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

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
                x=torch.FloatTensor(graph.x[:, :]),
                edge_index=torch.LongTensor(graph.edge_index),
                edge_attr=torch.FloatTensor(graph.edge_attr),
                smiles=graph.smiles,
                y=torch.FloatTensor(data[graph.smiles]),
                gas_spectra=torch.FloatTensor(gas_spectra[graph.smiles]),
                liquid_spectra=torch.FloatTensor(liquid_spectra[graph.smiles]),
                phase=args.phase
            )
        )

    return clean_graphs