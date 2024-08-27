source activate nbo
export PYTHONPATH=$PYTHONPATH:$(pwd)
python data/preprocess.py --size $1
python data/split.py --graphs_path data/graphs.pt --train_ratio $2 --val_ratio $3 --test_ratio $4 --from_NBO
source deactivate