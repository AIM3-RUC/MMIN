set -e
model_type=$1
run_idx=$2
gpu=$3


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap --model=cnn_audio --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--A_feat_type=melspec --model_type=$model_type --output_dim=4 
--niter=20 --niter_decay=30 --verbose --beta1=0.9
--batch_size=128 --lr=5e-4 --run_idx=$run_idx
--name=cnn --suffix={model_type}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done