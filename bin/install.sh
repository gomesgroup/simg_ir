conda create --name nbo python=3.8
source activate nbo
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=10.2 -c pytorch -c nvidia
conda install pyg=2.0 -c pyg
conda install -c conda-forge pytorch-lightning=1.7
pip uninstall torchmetrics
pip install torchmetrics==0.9.1
conda install -c conda-forge openbabel
pip install class-resolver
