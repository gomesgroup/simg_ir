import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import sys

# Configure Flask to use templates from webapp/templates
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Global variables
model = None
test_subset = None
x_axis = None

# Load model and test set
def load_model():
    global model, test_subset, x_axis
    
    dataset = "gas_649_max_sigma5_int"
    # Update paths to work from main directory
    checkpoint = torch.load(f"checkpoints/{dataset}/best_model.pt")
    
    # No need to append to path when running from main directory
    # Just make sure experiments is in the path
    if "experiments" not in sys.path:
        sys.path.append('.')
    from experiments.downstream_model.model import GNN

    model = GNN(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Update dataset path to work from main directory
    dataset_path = os.path.join("data/datasets", dataset)
    graphs = torch.load(os.path.join(dataset_path, "graphs.pt"), weights_only=False)

    test_indices = torch.load(os.path.join(dataset_path, "split", "test_indices.pt"), weights_only=False)
    test_subset = Subset(graphs, test_indices)
    x_axis = torch.arange(400, 4000, step=4)

# Initialize model before first request
@app.before_request
def before_request():
    global model
    if model is None:
        load_model()

# Function to calculate node importance using gradients
def calculate_node_importance(model, graph_data, wavenumber_idx):
    # Create a copy of the graph data as a single graph
    x = graph_data.x.clone().detach().requires_grad_(True)
    edge_index = graph_data.edge_index.clone().detach()
    edge_attr = graph_data.edge_attr.clone().detach()
    
    # Create a dummy batch index tensor
    batch_idx = torch.zeros(x.size(0), dtype=torch.long)
    
    # Forward pass
    model.zero_grad()
    output = model(x, edge_index, edge_attr, batch_idx)
    
    # Create a one-hot target for the specific wavenumber
    target = torch.zeros_like(output)
    target[0, wavenumber_idx] = 1.0
    
    # Calculate gradient w.r.t the specific wavenumber
    output[0, wavenumber_idx].backward(retain_graph=True)
    
    # Get gradients of input nodes
    node_importance = torch.abs(x.grad).sum(dim=1)
    
    return node_importance.detach().cpu().numpy()

# Function to generate visualization and return as base64 encoded image
def generate_visualization(graph_idx, wavenumber, top_n_nodes=5):
    # Get the graph from the test set
    graph_data = test_subset[graph_idx]
    
    # Create a copy for single graph prediction
    single_graph = graph_data.clone()
    
    # Run model prediction
    model.eval()
    with torch.no_grad():
        single_batch = torch.zeros(single_graph.x.size(0), dtype=torch.long)
        pred = model(single_graph.x, single_graph.edge_index, single_graph.edge_attr, single_batch)
    
    # Get prediction and ground truth
    prediction = pred.detach().cpu().numpy().flatten()
    ground_truth = graph_data.y.detach().cpu().numpy().flatten()
    
    # Extract data from the graph
    edge_index = graph_data.edge_index.cpu().numpy()
    node_features = graph_data.x.cpu().numpy()
    smiles = graph_data.smiles
    
    # Create networkx graph
    G = nx.Graph()
    
    # Get 2D coordinates using RDKit
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    
    # Assume the first N nodes are atoms
    num_atoms = mol.GetNumAtoms()
    
    # Build connectivity map
    connectivity = {}
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src not in connectivity:
            connectivity[src] = []
        if dst not in connectivity:
            connectivity[dst] = []
        connectivity[src].append(dst)
        connectivity[dst].append(src)
    
    # Add atom nodes with their element types
    atom_elements = {}
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        element = atom.GetSymbol()
        label = element
        pos = mol.GetConformer().GetAtomPosition(i)
        position = (pos.x, pos.y)
        G.add_node(i, pos=position, node_type='atom', label=label, element=element)
        atom_elements[i] = element
    
    # Calculate scale factor
    positions = np.array([(mol.GetConformer().GetAtomPosition(i).x, 
                         mol.GetConformer().GetAtomPosition(i).y) 
                        for i in range(num_atoms)])
    x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
    y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
    scale = max(x_range, y_range) * 0.08
    
    # First identify all atoms that should have lone pairs
    # and determine how many lone pairs they should have
    lonepair_atoms = {}
    
    # Count connected atoms and determine lone pairs based on element valence
    for i in range(num_atoms):
        element = atom_elements[i]
        atom = mol.GetAtomWithIdx(i)
        # Get the number of connected atoms (valence)
        connected_count = len([j for j in connectivity.get(i, []) if j < num_atoms])
        
        # Determine lone pairs based on chemistry rules
        lone_pairs = 0
        if element == 'O':  # Oxygen typically has 2 lone pairs
            lone_pairs = 2
        elif element == 'N':  # Nitrogen typically has 1 lone pair
            if connected_count <= 3:  # Only if not fully bonded (like NH4+)
                lone_pairs = 1
        elif element == 'F' or element == 'Cl' or element == 'Br' or element == 'I':  # Halogens
            lone_pairs = 3
        elif element == 'S':  # Sulfur can have lone pairs
            if connected_count <= 2:
                lone_pairs = 2
            elif connected_count <= 4:
                lone_pairs = 1
        
        if lone_pairs > 0:
            lonepair_atoms[i] = lone_pairs
    
    print(f"Atoms with lone pairs: {lonepair_atoms}")
    
    # Find non-atom nodes that should be treated as bond nodes
    bond_nodes = []
    for i in range(num_atoms, node_features.shape[0]):
        connected_atoms = [j for j in connectivity.get(i, []) if j < num_atoms]
        if len(connected_atoms) >= 2:
            bond_nodes.append(i)
    
    # Now find which non-atom nodes are not bond nodes and assign them as lone pairs to atoms
    available_nodes = [i for i in range(num_atoms, node_features.shape[0]) if i not in bond_nodes]
    assigned_lone_pairs = {}  # Map from atom index to list of lone pair node indices
    
    # Function to get directly connected non-atom nodes for an atom
    def get_connected_nonbond_nodes(atom_idx):
        return [n for n in connectivity.get(atom_idx, []) 
                if n >= num_atoms and n not in bond_nodes]
    
    # Assign available nodes as lone pairs to atoms that need them
    for atom_idx, num_lone_pairs in lonepair_atoms.items():
        # First try to use nodes that are already connected to this atom
        connected_nodes = get_connected_nonbond_nodes(atom_idx)
        
        if connected_nodes:
            # Use up to the required number of connected nodes
            assigned_lone_pairs[atom_idx] = connected_nodes[:num_lone_pairs]
            # Remove used nodes from available nodes
            for node in assigned_lone_pairs[atom_idx]:
                if node in available_nodes:
                    available_nodes.remove(node)
        else:
            # No connected nodes, assign new ones from available nodes
            assigned_pairs = []
            for _ in range(min(num_lone_pairs, len(available_nodes))):
                if available_nodes:
                    node = available_nodes.pop(0)
                    assigned_pairs.append(node)
            assigned_lone_pairs[atom_idx] = assigned_pairs
    
    print(f"Assigned lone pairs: {assigned_lone_pairs}")
    
    # Now position bond nodes and lone pair nodes
    # First position bonds
    for i in bond_nodes:
        connected_atoms = [j for j in connectivity.get(i, []) if j < num_atoms]
        valid_atoms = [j for j in connected_atoms if j < mol.GetNumAtoms()]
        
        if valid_atoms:
            atom_positions = [mol.GetConformer().GetAtomPosition(int(j)) for j in valid_atoms]
            position = (np.mean([p.x for p in atom_positions]), 
                      np.mean([p.y for p in atom_positions]))
        else:
            position = (0, 0)
        
        G.add_node(i, pos=position, node_type='bond', label='')
    
    # Then position lone pairs
    for atom_idx, lone_pair_nodes in assigned_lone_pairs.items():
        if not lone_pair_nodes:
            continue
            
        atom_pos = mol.GetConformer().GetAtomPosition(int(atom_idx))
        atom_point = np.array([atom_pos.x, atom_pos.y])
        
        # Get connected atoms to determine angle constraints
        connected_atoms = [j for j in connectivity.get(atom_idx, []) if j < num_atoms]
        
        # Create vectors from atom to its connected atoms
        vectors = []
        for connected_atom in connected_atoms:
            connected_pos = mol.GetConformer().GetAtomPosition(int(connected_atom))
            conn_point = np.array([connected_pos.x, connected_pos.y])
            vectors.append(conn_point - atom_point)
        
        # Determine available angles for lone pairs
        # Start with all angles being available
        all_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # 12 possible positions
        
        # Mark angles near bonds as unavailable
        available_angles = list(all_angles)
        for v in vectors:
            if np.linalg.norm(v) > 0:
                bond_angle = np.arctan2(v[1], v[0])
                # Mark angles within 30 degrees of bond as unavailable
                for angle in all_angles:
                    angle_diff = min(abs(angle - bond_angle), abs(angle - bond_angle + 2*np.pi), abs(angle - bond_angle - 2*np.pi))
                    if angle_diff < np.pi/6:  # 30 degrees
                        if angle in available_angles:
                            available_angles.remove(angle)
        
        # If no angles available, use all angles
        if not available_angles:
            available_angles = list(all_angles)
        
        # Sort available angles to distribute lone pairs evenly
        available_angles.sort()
        
        # Place lone pairs at available angles
        element = atom_elements[atom_idx]
        if element in ['O', 'S']:  # For oxygen and sulfur, prefer angles that are opposite each other
            if len(lone_pair_nodes) == 2 and len(available_angles) >= 2:
                # Find two angles that are approximately 180 degrees apart
                best_angle_pair = None
                best_diff = float('inf')
                
                for i, angle1 in enumerate(available_angles):
                    for j, angle2 in enumerate(available_angles[i+1:], i+1):
                        angle_diff = abs(abs(angle1 - angle2) - np.pi)
                        if angle_diff < best_diff:
                            best_diff = angle_diff
                            best_angle_pair = (angle1, angle2)
                
                if best_angle_pair:
                    available_angles = list(best_angle_pair)
        
        # Place lone pairs using available angles
        distance_factor = scale * 0.6  # Shorter distance for lone pairs
        
        for i, node_idx in enumerate(lone_pair_nodes):
            if i < len(available_angles):
                angle = available_angles[i]
                offset_x = distance_factor * np.cos(angle)
                offset_y = distance_factor * np.sin(angle)
                position = (atom_pos.x + offset_x, atom_pos.y + offset_y)
                print(f"Placing lone pair {node_idx} for atom {atom_idx} at angle {angle}, position {position}")
            else:
                # Fallback if we run out of available angles
                angle = 2 * np.pi * i / len(lone_pair_nodes)
                offset_x = distance_factor * np.cos(angle)
                offset_y = distance_factor * np.sin(angle)
                position = (atom_pos.x + offset_x, atom_pos.y + offset_y)
            
            G.add_node(node_idx, pos=position, node_type='lone_pair', label='LP')
    
    # Add any remaining nodes (that weren't assigned to bonds or lone pairs)
    for i in range(num_atoms, node_features.shape[0]):
        if i not in G.nodes():
            # Default position at center of graph
            avg_x = np.mean([pos[0] for pos in positions])
            avg_y = np.mean([pos[1] for pos in positions])
            G.add_node(i, pos=(avg_x, avg_y), node_type='other', label='?')
    
    # Add atom-atom edges
    atom_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src < num_atoms and dst < num_atoms:
            G.add_edge(src, dst)
            atom_edges.append((src, dst))
    
    # Get positions for drawing
    pos = nx.get_node_attributes(G, 'pos')
    
    # Function to get index from wavenumber
    def wavenumber_to_index(wn):
        return int((wn - 400) / 4)
    
    # Convert wavenumber to index for the model
    wn_idx = wavenumber_to_index(wavenumber)
    
    # Calculate node importance using gradients
    node_importance = calculate_node_importance(model, graph_data, wn_idx)
    
    # Get top influential nodes
    top_indices = np.argsort(np.abs(node_importance))[-top_n_nodes:]
    
    # Create a new figure for the molecular graph
    fig = plt.figure(figsize=(10, 8))
    ax_mol = fig.add_subplot(1, 1, 1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax_mol, edgelist=atom_edges, 
                          alpha=0.8, width=1.5, edge_color='blue')
    
    # Get node lists by type
    atom_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'atom']
    bond_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'bond']
    lone_pair_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'lone_pair']
    
    # Define colors and sizes
    atom_color = '#1f78b4'  # Blue
    bond_color = '#33a02c'  # Green
    lone_pair_color = '#ff7f00'  # Orange
    
    atom_size = 700
    bond_size = 300
    lp_size = 250
    
    # Create color and size lists
    atom_colors = [atom_color] * len(atom_nodes)
    bond_colors = [bond_color] * len(bond_nodes)
    lp_colors = [lone_pair_color] * len(lone_pair_nodes)
    
    atom_sizes = [atom_size] * len(atom_nodes)
    bond_sizes = [bond_size] * len(bond_nodes)
    lp_sizes = [lp_size] * len(lone_pair_nodes)
    
    # Update colors and sizes for influential nodes
    highlight_color = 'red'
    size_multiplier = 1.5
    
    for idx in top_indices:
        if idx in atom_nodes:
            node_pos = atom_nodes.index(idx)
            atom_colors[node_pos] = highlight_color
            atom_sizes[node_pos] = atom_size * size_multiplier
        elif idx in bond_nodes:
            node_pos = bond_nodes.index(idx)
            bond_colors[node_pos] = highlight_color
            bond_sizes[node_pos] = bond_size * size_multiplier
        elif idx in lone_pair_nodes:
            node_pos = lone_pair_nodes.index(idx)
            lp_colors[node_pos] = highlight_color
            lp_sizes[node_pos] = lp_size * size_multiplier
    
    # Draw nodes with highlight effect
    nx.draw_networkx_nodes(G, pos, ax=ax_mol, nodelist=atom_nodes, 
                          node_color=atom_colors, node_size=atom_sizes, 
                          alpha=0.9, node_shape='o')
    
    nx.draw_networkx_nodes(G, pos, ax=ax_mol, nodelist=bond_nodes, 
                          node_color=bond_colors, node_size=bond_sizes, 
                          alpha=0.8, node_shape='s')
    
    nx.draw_networkx_nodes(G, pos, ax=ax_mol, nodelist=lone_pair_nodes, 
                          node_color=lp_colors, node_size=lp_sizes, 
                          alpha=0.8, node_shape='d')
    
    # Add labels only to atoms for clarity
    atom_labels = {n: G.nodes[n]['label'] for n in atom_nodes}
    nx.draw_networkx_labels(G, pos, ax=ax_mol, labels=atom_labels, font_size=12, font_color='black')
    
    # Add a legend
    atom_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=atom_color, 
                           markersize=15, label='Atom')
    bond_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=bond_color, 
                           markersize=10, label='Bond')
    lp_patch = plt.Line2D([0], [0], marker='d', color='w', markerfacecolor=lone_pair_color, 
                         markersize=10, label='Lone Pair')
    highlight_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                               markersize=15, label=f'Top {top_n_nodes} Influential Nodes')
    
    ax_mol.legend(handles=[atom_patch, bond_patch, lp_patch, highlight_patch], 
                loc='upper right', frameon=True)
    
    ax_mol.set_title(f"Molecule: {smiles}\nWavenumber: {wavenumber} cm⁻¹", fontsize=14)
    ax_mol.axis('off')
    
    # Convert plot to base64 encoded image
    buf = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    plt.close(fig)
    
    # Encode the image to base64 string
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    # Prepare graph data for potential client-side rendering
    graph_data_for_client = {
        'nodes': [],
        'edges': []
    }
    
    # Add nodes with their attributes
    for node in G.nodes():
        node_data = {
            'id': int(node),
            'type': G.nodes[node]['node_type'],
            'label': G.nodes[node]['label'],
            'x': float(pos[node][0]),
            'y': float(pos[node][1]),
            'importance': float(node_importance[node])
        }
        
        # Add highlighted status
        if node in top_indices:
            node_data['highlighted'] = True
        
        graph_data_for_client['nodes'].append(node_data)
    
    # Add edges
    for edge in G.edges():
        graph_data_for_client['edges'].append({
            'source': int(edge[0]),
            'target': int(edge[1])
        })
    
    # Return both image-based and data-based representations
    return {
        'molecule_image': img_data,
        'smiles': smiles,
        'spectrum_data': {
            'x_axis': x_axis.tolist(),
            'prediction': prediction.tolist(),
            'ground_truth': ground_truth.tolist()
        },
        'graph_data': graph_data_for_client,
        'node_importance': node_importance.tolist()
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html', graph_count=len(test_subset))

@app.route('/get_visualization', methods=['POST'])
def get_visualization():
    graph_idx = int(request.form.get('graph_idx', 0))
    wavenumber = int(request.form.get('wavenumber', 1500))
    top_n_nodes = int(request.form.get('top_n_nodes', 5))
    
    result = generate_visualization(graph_idx, wavenumber, top_n_nodes)
    
    return jsonify({
        'image': result['molecule_image'],
        'smiles': result['smiles'],
        'wavenumber': wavenumber,
        'graph_idx': graph_idx,
        'spectrum_data': result['spectrum_data'],
        'graph_data': result['graph_data'],
        'node_importance': result['node_importance']
    })

@app.route('/get_graph_count')
def get_graph_count():
    return jsonify({'count': len(test_subset)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
