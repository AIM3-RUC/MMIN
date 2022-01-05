# MMIN

This repo implements the Missing Modality Imagination Network(MMIN) for the following paper:
"Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities" 

# Environment

``` 
python 3.7.0
pytorch >= 1.0.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

The preprocess of feature was done handcrafted in several steps, we will make it a automatical running script in the next update. You can download the preprocessed feature to run the code.

+ For Training MMIN on IEMOCAP:

    First training a model fusion model with all audio, visual and lexical modality as the pretrained encoder.

    ```bash
    bash scripts/CAP_utt_fusion.sh AVL [num_of_expr] [GPU_index]
    ```

    Then

    ```bash
    bash scripts/CAP_mmin.sh [num_of_expr] [GPU_index]
    ```

+ For Training MMIN on MSP-improv: 

    ```bash
    bash scripts/MSP_utt_fusion.sh AVL [num_of_expr] [GPU_index]
    ```

    ```
    bash scripts/MSP_mmin.sh [num_of_expr] [GPU_index]
    ```

Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get_opt.py and the ```modify_commandline_options``` method of each model you choose.

# Download the features
Baidu Yun Link
IEMOCAP A V L modality Features
链接: https://pan.baidu.com/s/1WmuqNlvcs5XzLKfz5i4iqQ 提取码: gn6w 

# License
MIT license. 

Copyright (c) 2021 AIM3-RUC lab, School of Information, Renmin University of China.

# Citation
If you find our paper and this code usefull, please consider cite
```
@inproceedings{zhao2021missing,
  title={Missing modality imagination network for emotion recognition with uncertain missing modalities},
  author={Zhao, Jinming and Li, Ruichen and Jin, Qin},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={2608--2618},
  year={2021}
}
```
