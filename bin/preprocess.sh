source activate nbo
export PYTHONPATH=$PYTHONPATH:$(pwd)
python data/preprocess.py --size $1
conda deactivate