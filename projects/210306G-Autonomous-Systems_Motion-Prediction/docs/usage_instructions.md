# Student Usage Instructions

Set up conda environment and clone the github repo ( based on https://github.com/yuyangw/MolCLR )

```
# create a new environment
$ conda create --name molclr python=3.7
$ conda activate molclr

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of MolCLR
$ git clone https://github.com/JaninduTharuka/MolCLR.git
$ cd MolCLR
```

download datasets from [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing).
extract zip file under ./data folder.

change parameters in `config_finetune.yaml` and run,
```
$ python finetune.py
```
