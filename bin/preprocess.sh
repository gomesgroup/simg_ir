source activate nbo
export PYTHONPATH=$PYTHONPATH:$(pwd)
python data/preprocess.py --filename data/simulated/$1.json --graphs_path data/datasets/$1/graphs.pt --from_NBO
source deactivate