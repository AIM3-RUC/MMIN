set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=multimodal --model=impl_fusion
--gpu_ids=$gpu --trn_modality=A --test_modality=A 
--weight_a=1.0 --weight_v=0.0 --weight_l=0.5 --embd_size=128
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--A_type=wav2vec_raw_3mean --input_dim_a=768 --norm_method=trn --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --corpus_name=IEMOCAP
--output_dim=4 --cls_layers=128,128,128 --dropout_rate=0.3
--niter=10 --niter_decay=10 --verbose --beta1=0.9 
--batch_size=128 --lr=2e-4 --run_idx=$run_idx
--name=CAP_impl_fusion --suffix=trn{trn_modality}_test{test_modality}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done