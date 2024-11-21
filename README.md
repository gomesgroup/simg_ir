# SIMG-IR: Stereochemical Graph-based IR Spectrum Prediction

A deep learning framework for predicting infrared (IR) spectra using stereochemical molecular graph representations.

## Overview

SIMG-IR leverages graph neural networks to predict IR spectra for:
- Single molecules
- Multi-component molecular mixtures

The model uses a specialized stereochemical graph representation that captures 3D molecular structure and connectivity information critical for accurate IR spectrum prediction.

## Key Features

- Stereochemical graph construction from molecular structures
- Graph neural network architecture optimized for spectral prediction
- Support for both individual molecules and molecular mixtures
- High-resolution IR spectra prediction (4 cm⁻¹ resolution)
- Parallel training across multiple GPUs

## Usage

The pipeline consists of:
1. Preprocessing molecular data into stereochemical graphs
2. Training the GNN model
3. Predicting IR spectra for new molecules

See documentation for detailed usage instructions.