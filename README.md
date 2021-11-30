# Coordinated Proximal Policy Optimization

This repository implements Coordinated Proximal Policy Optimization (CoPPO),  accompanying paper "[Coordinated Proximal Policy Optimization](https://arxiv.org/abs/2111.04051)". 

## Requirements

Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install the package
cd CoPPO
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

## Train

```cmd
cd onpolicy/scripts
chmod +x ./train_smac.sh
./train_smac.sh
```

Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [document](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Weights & Biases instead of Tensorboard. 



##  Note

While heavily based on the implementation of [MAPPO](https://github.com/marlbenchmark/on-policy), we use `concatenate(s, o)` ($s$ and $o$ are provided by the original SMAC environment)  rather than the Feature-Pruned Agent-Specific Global State mentioned in the MAPPO [paper](https://arxiv.org/abs/2103.01955) as the input of value functions  (i.e.  `share_obs` in the code).













