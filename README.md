# CaRTS: Causality-driven Robot Tool Segmentation from Vision and Kinematics Data
This repo hosts the code for implementing the CaRTS algorithms for Robot Tool segmentation.

> [**CaRTS: Causality-driven Robot Tool Segmentation from Vision and Kinematics Data**](https://arxiv.org/abs/2203.09475),            
> Hao Ding, Jintan Zhang, Peter Kazanzides, Jie Ying Wu, Mathias Unberath 
> In: Proc. MICCAI, 2022  
> *arXiv preprint ([arXiv 2203.09475](https://arxiv.org/abs/2203.09475))*  

![eye_candy](img/eye_candy.jpeg)

![causal_model](img/causal_model.jpeg)

![CaRTS](img/CaRTS.jpeg)

## Highlights
- **Complementary Causal Model for Robot Tool Segmentation**  
- **Architecture CaRTS, based on the causal model** 

## Installation


### We created an environment.yml for creating the exact same conda envrionment that we ran the code on,You can simply install the environment by this conda command:

    conda env create -f environment.yml

### Our CUDA VERSION is:
    
    Cuda compilation tools, release 11.6, V11.6.124

### Our GPU DRIVER VERSION is:

    510.60.02


## Usage

We only used one GPU for training and inference so we haven't implement multi-gpu version.

### To run training, find the right name for the config you want in the ![file](config/__init__.py):

python train_nn.py <config_name>

### for example:

python train_nn.py CaRTSACSCTS

### To run inference, give the name of the config and the path to the checkpoint file for networks to load:

python test.py <config_name> <path_to_checkpoint>

### for example:

python test.py CaRTSACSCTS ../CaRTS/checkpoints/carts_base_cts/model_49.pth

## Dataset preparation:

We are working on make a more comprehensive version of the causal tool segmentation dataset. If you need the dataset that is used in this paper, please contact Hao Ding ([email](mailto:hding15@jhu.edu)) and Mathias Unberath([email](mailto:unberath@jhu.edu)）.

If you want to use your own dataset please write your own dataloader with the same format that the files in the dataset folder has.



## Citations
Please consider citing our papers in your publications if this repo helps you. 
```
@inproceedings{ding2022carts,
  title     =  {CaRTS: Causality-driven Robot Tool Segmentation from Vision and Kinematics Data},
  author    =  {Ding, Hao and Zhang, Jintan and Kazanzides, Peter and Wu, Jie Ying and Unberath, Mathias},
  booktitle =  {Proc. MICCAI},
  year      =  {2022}
}
```

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact Hao Ding ([email](mailto:hding15@jhu.edu)) and Mathias Unberath([email](mailto:unberath@jhu.edu)）
