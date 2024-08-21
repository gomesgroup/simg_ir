export PYTHONPATH=$PYTHONPATH:$(pwd)
source .venv/bin/activate
python experiments/downstream_model/train.py --graphs_path outputs/graphs.pt --bs 64 --model_config experiments/downstream_model/model_config_GCN_tg.yaml