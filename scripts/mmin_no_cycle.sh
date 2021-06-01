set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=mmin_no_cycle 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=$gpu
--A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size_l=128
--AE_layers=128,64,32 --n_blocks=5
--pretrained_path="checkpoints/utt_fusion_AVL_run2"
--ce_weight=1.0 --mse_weight=0.2 --cycle_weight=0.2
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.2
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type kaiming
--batch_size=128 --lr=1e-3 --run_idx=$run_idx
--name=mmin_no_cycle --suffix=block_{n_blocks}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done