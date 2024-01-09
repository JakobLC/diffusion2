This is a repository for the beginning of a PhD by Jakob Lønborg Christensen at The Technical University of Denmark (DTU).

To create a conda environment for the repository run the following in a terminal with Anaconda (or preferably mamba) installed:
```
mamba create -n diff-env python=3.8.12
mamba activate diff-env
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -n diff-env -c conda-forge --file requirements.txt
pip install git+https://github.com/JakobLC/jlc.git --no-deps
```

Afterwards, training and sampling can respectively be done by running:
```
python train.py
python sample.py
```

