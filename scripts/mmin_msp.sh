set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 12`;
do

cmd="python train_miss.py --dataset_mode=msp_miss --model=mmin 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=$gpu
--input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=1024 --embd_size_l=128
--AE_layers=256,128,64 --n_blocks=5
--pretrained_path="checkpoints/MSP_fusion_AVL_run2"
--ce_weight=1.0 --mse_weight=2.0 --cycle_weight=0.5
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.5
--niter=30 --niter_decay=30 --verbose --init_type kaiming
--batch_size=256 --lr=1e-3 --run_idx=$run_idx
--name=mmin_MSP_f1 --suffix=block_{n_blocks}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done