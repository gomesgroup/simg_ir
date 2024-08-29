source activate nbo
export PYTHONPATH=$PYTHONPATH:$(pwd)
python data/preprocess.py --graphs_path data/graphs.pt --from_NBO --size $1
source deactivate