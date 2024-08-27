export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python data/split.py --graphs_path data/graphs.pt --train_ratio $1 --val_ratio $2 --test_ratio $3 --from_NBO
deactivate