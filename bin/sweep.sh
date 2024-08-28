export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python experiments/downstream_model/train_parallel.py --graphs_path data/nice_graphs.pt --split_dir data/split --model_config configs/model_config_GCN_tg.yaml --max_epochs 10 --bs 32 
deactivate