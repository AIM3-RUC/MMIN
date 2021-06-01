# MMIN

This repo implements the Missing Modality Imagination Network(MMIN) for the following paper:
"Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities" 

# Environment

``` 
pytorch >= 1.0.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```

For Training MMIN on IEMOCAP:
```bash
bash scripts/mmin.sh [num_of_expr] [GPU_index]
```

For Training MMIN on MSP-improv: 
```
bash scripts/mmin_msp.sh [num_of_expr] [GPU_index]
```

Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/train_opt.py and the ```modify_commandline_options``` method of each model you choose.

# License

# Citation
