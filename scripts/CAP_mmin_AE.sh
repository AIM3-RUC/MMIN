set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=mmin_AE
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu
--A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size_l=128 
--AE_layers=512,450,384,256 --num_thread=0 --corpus=IEMOCAP 
--pretrained_path='checkpoints/CAP_utt_fusion_AVL_run1'
--ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0
--output_dim=4 --cls_layers=128,128 --init_type=normal --dropout_rate=0.5
--niter=20 --niter_decay=30 --verbose --print_freq=10
--batch_size=256 --lr=1e-3 --run_idx=$run_idx --weight_decay=1e-5
--name=mmin_IEMOCAP_AE --suffix=run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done