This is a repository for the master's thesis titled "Diffusion Models for Image Segmentation" by Jakob Lønborg Christensen at The Technical University of Denmark (DTU).

To create a conda environment for the repository run the following in a terminal with Anaconda installed:
```
conda create -n diff-env python=3.8.12
conda activate diff-env
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/JakobLC/jlc.git
```

Afterwards, training and sampling can respectively be done by running:
```
python train.py
python sample.py
```
The repository is largely inspired by https://github.com/tomeramit/SegDiff

```
mamba create -n diff-env python=3.8.12
mamba activate diff-env
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -n diff-env -c conda-forge --file requirements.txt
pip install git+https://github.com/JakobLC/jlc.git --no-deps
```
