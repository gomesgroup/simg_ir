# Molecular Graph Visualization Web App

This Flask web application allows you to visualize molecular graphs and highlight nodes with the biggest effect at specific wavenumber positions using a trained Graph Neural Network model.

## Features

- Interactive wavenumber selection with a slider
- Visualization of molecular graphs with node highlighting
- Display of IR spectrum prediction vs ground truth
- Selection of different molecules from the test set
- Adjustable number of top influential nodes to highlight

## Requirements

- Python 3.7+
- Flask
- PyTorch
- PyTorch Geometric
- RDKit
- NetworkX
- Matplotlib
- NumPy

## Installation

1. Make sure you have all the required packages installed:

```bash
pip install flask torch torch_geometric rdkit networkx matplotlib numpy
```

2. Ensure your model and dataset paths are correct in the `webapp/app.py` file.

## Running the Application

### Running from the main directory (recommended)

1. From the main project directory (simg_ir), run:

```bash
python run_webapp.py
```

2. Open a web browser and go to:

```
http://127.0.0.1:5002/
```

### Running from the webapp directory

Alternatively, you can still run the app from the webapp directory:

1. Navigate to the webapp directory:

```bash
cd webapp
```

2. Run the Flask application:

```bash
python app.py
```

3. Open a web browser and go to:

```
http://127.0.0.1:5002/
```

## Usage

- Use the "Molecule Index" input to select a different molecule from the test set
- Adjust the wavenumber slider to see how different wavenumbers affect node importance
- Change the "Top Influential Nodes" value to highlight more or fewer nodes
- Click "Update Visualization" to apply your changes

## How It Works

1. The server loads a pre-trained GNN model for IR spectrum prediction
2. When you select a wavenumber, the app calculates which nodes in the molecular graph have the biggest effect on the prediction at that wavenumber position using gradient-based attribution
3. The visualization highlights these influential nodes in red and displays the IR spectrum with a vertical line at the selected wavenumber 