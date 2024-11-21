export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python data/split.py --graphs_dir data/datasets/$1 --train_ratio $2 --val_ratio $3 --test_ratio $4 --split_type $5
deactivate