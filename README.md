# SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions

This repo hosts the code for implementing the baseline algorithms for SegSTRONG-C.

![eye_candy](img/eye_candy_2.png)

This challenge is originated from:
> [**CaRTS: Causality-driven Robot Tool Segmentation from Vision and Kinematics Data**](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_37),            
> Hao Ding, Jintan Zhang, Peter Kazanzides, Jie Ying Wu, Mathias Unberath 
> Proc. MICCAI, 2022  
> *arXiv preprint ([arXiv 2203.09475](https://arxiv.org/abs/2203.09475))*
>
> [**Rethinking Causality-driven Robot Tool Segmentation with Temporal Constraints**](https://link.springer.com/article/10.1007/s11548-023-02872-8),
> Hao Ding, Jie Ying Wu, Zhaoshuo Li, Mathias Unberath
> Int J CARS 18, 1009–1016 (2023)
> *arXiv preprint ([arXiv 2203.09475](https://arxiv.org/abs/2212.00072))*
>
> [**SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions -- An EndoVis'24 Challenge**](https://arxiv.org/abs/2407.11906),
> Hao Ding, Tuxun Lu, Yuqian Zhang, Ruixing Liang, Hongchao Shu, Lalithkumar Seenivasan, Yonghao Long, Qi Dou, Cong Gao, Mathias Unberath
> 2024
> *arXiv preprint ([arXiv 2203.09475](https://arxiv.org/abs/2407.11906))*

## Installation

### We provided docker for easy installation, the environment can be easily set up via:

    cd docker
    docker build ./ -t segstrongc:latest
    docker run --rm -v "LOCAL_DATADIR":/workspace/data --gpus='"device={GPU_IDS}"' -it segstrongc:latest


## Usage

We only used one GPU for training and inference so we haven't implement multi-gpu version.

### To run training, find the right name for the config you want in the [file](config/__init__.py):

    python train.py --config CONFIG_FILENAME

### for example:

    python train.py --config UNet_SegSTRONGC

### To run inference on validation set, give the name of the config and the path to the checkpoint file for networks to load:

    python validate.py --config CONFIG_FILENAME --model_path CHECKPOINT_PATH --domain DOMAIN_NAME

### for example:

    python validate.py --config UNet_SegSTRONGC --model_path checkpoints/unet_segstrongc/model_39.pth --domain regular

### The final test will be on test set(for example):

    python validate.py --config UNet_SegSTRONGC --model_path checkpoints/unet_segstrongc/model_39.pth --test True --domain smoke --save_dir /workspace/data/SegSTRONG-C/results/smoke

## Dataset preparation:

Please refer to our ([website](segstrongc.cs.jhu.edu)) for registration and data downloading

## Citations
Please consider citing our papers in your publications if this repo helps you. 
```
@inproceedings{ding2022carts,
  title={CaRTS: Causality-Driven Robot Tool Segmentation from Vision and Kinematics Data},
  author={Ding, Hao and Zhang, Jintan and Kazanzides, Peter and Wu, Jie Ying and Unberath, Mathias},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={387--398},
  year={2022},
  organization={Springer}
}

@article{Ding2022RethinkingCR,
  title={Rethinking causality-driven robot tool segmentation with temporal constraints},
  author={Hao Ding and Jie Ying Wu and Zhaoshuo Li and M. Unberath},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2022},
  pages={1009 - 1016},
}

@misc{ding2024segstrongcsegmentingsurgicaltools,
      title={SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions -- An EndoVis'24 Challenge}, 
      author={Hao Ding and Tuxun Lu and Yuqian Zhang and Ruixing Liang and Hongchao Shu and Lalithkumar Seenivasan and Yonghao Long and Qi Dou and Cong Gao and Mathias Unberath},
      year={2024},
      eprint={2407.11906},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11906}, 
}
```

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact Hao Ding ([email](mailto:hding15@jhu.edu)) and Mathias Unberath([email](mailto:unberath@jhu.edu)）
