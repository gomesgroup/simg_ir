export PYTHONPATH=$PYTHONPATH:$(pwd)
source .venv/bin/activate
python experiments/downstream_model/train.py --graphs_path data/graphs.pt --max_epochs 1 --bs 128 --model_config configs/model_config_GCN_tg.yaml --from_NBO