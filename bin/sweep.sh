export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python experiments/downstream_model/train_parallel.py --graphs_path data/datasets/$1/graphs.pt --split_dir data/datasets/$1/split --model_config configs/model_config_GCN_tg.yaml --max_epochs 300 --bs 8 
deactivate