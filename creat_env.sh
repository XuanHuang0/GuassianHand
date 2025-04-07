conda env create -f environment.yml
conda activate GuassianHand
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install git+https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/amundra15/livehand.git
cd ./livehand
pip install --editable .
