import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from flask import Flask, render_template, jsonify, request
from experiments.downstream_model.model import GNN

# Create Flask app with the correct template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load data
dataset = "gas_649_max_sigma5_int"
dataset_path = os.path.join("data/datasets", dataset)
graphs = torch.load(os.path.join(dataset_path, "graphs.pt"), weights_only=False)
test_indices = torch.load(os.path.join(dataset_path, "split", "test_indices.pt"), weights_only=False)
test_subset = Subset(graphs, test_indices)

# Load model for predictions
checkpoint = torch.load(f"checkpoints/{dataset}/best_model.pt")
model = GNN(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create wavenumber range as specified
wavenumbers = torch.arange(400, 4000, step=4).tolist()

def process_graph(graph_idx=0):
    """Process a graph from the dataset and prepare it for visualization"""
    graph = graphs[graph_idx]
    G = to_networkx(graph)
    
    # Create a more structured representation of the molecular graph
    info = {
        "atom": [],       # List of atom objects with properties
        "bond": [],       # List of bond objects connecting atoms
        "lone_pair": [],  # List of lone pair objects
        "interaction": [] # List of interaction objects
    }

    lookup = {}

    # First pass: identify and create atoms
    for i in range(len(graph.symbol)):
        if graph.symbol[i] not in ["BND", "LP"]:
            d = {
                "idx": i,
                "type": "atom",
                "symbol": graph.symbol[i],
            }
            info["atom"].append(d)
            lookup[i] = d
            
    # Second pass: create bonds, lone pairs, and interactions
    atom_ids = [atom["idx"] for atom in info["atom"]]

    for i in range(len(graph.x)):
        if i < len(graph.symbol):
            if graph.symbol[i] == "BND":
                connected_atoms = list(set(G.neighbors(i)).intersection(set(atom_ids)))
                d = {
                    "idx": i,
                    "type": "bond",
                    "nbrs": connected_atoms
                }
                info["bond"].append(d)
                lookup[i] = d
            elif graph.symbol[i] == "LP":
                connected_atoms = list(set(G.neighbors(i)).intersection(set(atom_ids)))
                d = {
                    "idx": i,
                    "type": "lone_pair",
                    "nbrs": connected_atoms
                }
                info["lone_pair"].append(d)
                lookup[i] = d
        else:
            neighbors = list(G.neighbors(i))
            d = {
                "idx": i,
                "type": "interaction",
                "nbrs": neighbors
            }
            info["interaction"].append(d)
            lookup[i] = d

    # Create RDKit mol object from SMILES and add hydrogens
    mol = Chem.MolFromSmiles(graph.smiles)
    mol = Chem.AddHs(mol)

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    conf = mol.GetConformer()
    coords_2d = []

    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords_2d.append(np.array([pos.x, pos.y]))
        
    for atom_dict in info["atom"]:
        atom_dict["pos"] = coords_2d[atom_dict["idx"]].tolist()  # Convert numpy arrays to lists for JSON
        
    for bond_dict in info["bond"]:
        nbrs = bond_dict["nbrs"]
        nbr1 = nbrs[0]
        pos1 = lookup[nbr1]["pos"]
        nbr2 = nbrs[1]
        pos2 = lookup[nbr2]["pos"]
        bond_dict["pos"] = ((np.array(pos1) + np.array(pos2)) / 2).tolist()
        
    for lone_pair_dict in info["lone_pair"]:
        nbrs = lone_pair_dict["nbrs"]
        nbr = nbrs[0]
        pos = lookup[nbr]["pos"]
        # Add small Gaussian noise to position the lone pair slightly away from the atom
        noise = np.random.normal(0, 0.2, 2)  # Mean 0, std 0.2, for both x and y
        lone_pair_dict["pos"] = (np.array(pos) + noise).tolist()
        
    for interaction_dict in info["interaction"]:
        # Calculate average position of all neighbors
        if interaction_dict["nbrs"]:
            pos_sum = np.zeros(2)
            for nbr in interaction_dict["nbrs"]:
                if nbr in lookup and "pos" in lookup[nbr]:
                    pos_sum += np.array(lookup[nbr]["pos"])
            interaction_dict["pos"] = (pos_sum / len(interaction_dict["nbrs"])).tolist()
        else:
            # Fallback position if no neighbors with positions
            interaction_dict["pos"] = [0, 0]
    
    # Create edge data for visualization
    edges = []
    for node_type in ["bond", "lone_pair"]:
        for node in info[node_type]:
            for nbr in node["nbrs"]:
                edges.append({
                    "source": node["idx"],
                    "target": nbr
                })
    
    # Transform into nodes array for D3
    nodes = []
    for node_type in ["atom", "bond", "lone_pair"]:
        for node in info[node_type]:
            node_data = {
                "id": node["idx"],
                "type": node_type,
                "x": node["pos"][0],
                "y": node["pos"][1]
            }
            # Add type-specific properties
            if node_type == "atom":
                node_data["symbol"] = node["symbol"]
            elif node_type in ["bond", "lone_pair"]:
                node_data["neighbors"] = node["nbrs"]
            
            nodes.append(node_data)
    
    # Get ground truth spectrum from graph.y
    ground_truth = graph.y.tolist()
    
    # Get predicted spectrum with proper model call (batch size 1)
    with torch.no_grad():
        # Use proper model call with expected arguments
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, 
                          torch.zeros(graph.x.size(0), dtype=torch.long))  # Single graph batch
        prediction = prediction.squeeze().tolist()
    
    return {
        "nodes": nodes,
        "links": edges,
        "smiles": graph.smiles,
        "wavenumbers": wavenumbers,
        "ground_truth_spectrum": ground_truth,
        "predicted_spectrum": prediction
    }

def compute_node_importance(graph_idx, wavenumber_idx):
    """Compute the importance of each node for a specific wavenumber using gradients"""
    graph = graphs[graph_idx]
    
    # Clone graph data and enable gradient tracking
    x = graph.x.clone().detach().requires_grad_(True)
    edge_index = graph.edge_index.clone().detach()
    edge_attr = graph.edge_attr.clone().detach()
    batch = torch.zeros(x.size(0), dtype=torch.long)
    
    # Forward pass
    model.zero_grad()
    output = model(x, edge_index, edge_attr, batch)
    
    # Get the selected wavenumber's prediction
    target_pred = output.squeeze()[wavenumber_idx]
    
    # Compute gradients with respect to node features
    target_pred.backward()
    
    # Calculate importance scores based on gradient magnitude
    importance = torch.sum(torch.abs(x.grad), dim=1)
    importance_scores = importance.detach().cpu().numpy()
    
    # Get indices of top 5 most important nodes
    top_indices = np.argsort(importance_scores)[-5:][::-1].tolist()
    
    # Get the importance scores for the top indices
    top_scores = importance_scores[top_indices].tolist()
    
    return {
        "top_indices": top_indices,
        "top_scores": top_scores,
        "wavenumber": wavenumbers[wavenumber_idx]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph/<int:graph_id>')
def get_graph(graph_id):
    try:
        graph_data = process_graph(graph_id)
        return jsonify(graph_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/importance')
def get_node_importance():
    try:
        graph_id = int(request.args.get('graph_id', 0))
        wavenumber_idx = int(request.args.get('wavenumber_idx', 0))
        
        importance_data = compute_node_importance(graph_id, wavenumber_idx)
        return jsonify(importance_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/graphs/count')
def get_graphs_count():
    return jsonify({"count": len(graphs)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)