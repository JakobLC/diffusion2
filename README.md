This is a repository for the beginning of a PhD by Jakob Lønborg Christensen at The Technical University of Denmark (DTU).

To create a conda environment for the repository run the following in a terminal with Anaconda/Miniconda installed:
```
conda create -n diff-env python=3.8.12
conda activate diff-env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -n diff-env -c conda-forge --file requirements.txt
pip install git+https://github.com/JakobLC/jlc.git --no-deps
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```
SAM2 evaluation also requires you to clone `https://github.com/facebookresearch/segment-anything-2.git` and add it to the python path in `sam.py`.
Afterwards, training and sampling can respectively be done by running:
```
python train.py
python sample.py
```


To create the SAM2 env, run
```bash
conda create -n sam2-env python=3.12
conda activate sam2-env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-image=0.23.2 scikit-learn=1.5.1 pandas=2.2.2 scipy=1.14.1 jsonlines -c conda-forge
pip install git+https://github.com/JakobLC/jlc.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install git+https://github.com/openai/CLIP.git
pip install albumentations==1.4.15
pip install nibabel==5.2.1
pip install -r requirements_sam.txt
```
(should hopefully get pytorch 2.4.1)

