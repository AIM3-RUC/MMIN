set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=utt_dataaug
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --gpu_ids=$gpu
--A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size_l=128
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3 --corpus=IEMOCAP
--niter=20 --niter_decay=20 --init_type=normal --beta1=0.9
--batch_size=128 --lr=2e-4 --run_idx=$run_idx --modality=AVL
--name=IEMOCAP_data_aug --suffix=run{run_idx} --has_test --verbose
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done