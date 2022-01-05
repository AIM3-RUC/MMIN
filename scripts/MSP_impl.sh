set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 12`;
do

cmd="python train_baseline.py --dataset_mode=multimodal --corpus_name=MSP
--gpu_ids=$gpu --trn_modality=AL --test_modality=A --model=impl_fusion 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--A_type=comparE_raw  --input_dim_a=130 --norm_method=trn --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size=128
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=15 --niter_decay=15 --verbose --beta1=0.9
--batch_size=64 --lr=5e-4 --run_idx=$run_idx
--name=MSP_impl_fusion --suffix=trn_{trn_modality}_test{test_modality}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done