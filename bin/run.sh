export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/simg_ir/.venv/bin/activate
python experiments/downstream_model/train.py --dataset_dir data/dataset --model_config configs/model_config_GCN_tg.yaml --max_epochs 1 --bs 128 
deactivate