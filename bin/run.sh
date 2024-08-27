export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python experiments/downstream_model/train_parallel.py --dataset_dir data/dataset --model_config configs/model_config_GCN_tg.yaml --max_epochs 5 --bs 32 
deactivate