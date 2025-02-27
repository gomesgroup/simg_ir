export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python experiments/downstream_model/train_parallel.py --graphs_path data/datasets/$1/graphs.pt --split_dir data/datasets/$1/split --sweep_config data/datasets/$1/sweep_config.json --model_config configs/model_config_GCN_tg.yaml --max_epochs 100 --bs 16 --gpu_ids 3
deactivate