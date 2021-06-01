set -e
# test missing on pretrained AVL
# for cv in `seq 1 1 12`;do
# echo "Testing CV: $cv"
# CUDA_VISIBLE_DEVICES=4 python eval_miss.py \
#     --dataset_mode=msp_miss \
#     --model_path=MSP_baseline_AVL_run6 \
#     --checkpoints_dir=checkpoints \
#     --log_dir=logs \
#     --gpu_id=0 \
#     --name=eval_msp_F_baseline_run6 \
#     --cvNo=$cv \
#     --eval_miss \
#     --total_cv=12
# done

# test full on data augment baseline
# for cv in `seq 1 1 12`;do
# echo "Testing CV: $cv"
# CUDA_VISIBLE_DEVICES=0 python eval_miss.py \
#     --dataset_mode=msp_multimodal \
#     --model_path=MSP_data_aug_run3 \
#     --checkpoints_dir=checkpoints \
#     --log_dir=logs \
#     --gpu_id=0 \
#     --name=eval_msp_dataaug_on_full_run3 \
#     --cvNo=$cv \
#     --total_cv=12
# done

# test full on MMIN
for cv in `seq 1 1 10`;do
echo "Testing CV: $cv"
CUDA_VISIBLE_DEVICES=7 python eval_miss.py \
    --dataset_mode=multimodal \
    --model_path=mmin_block_5_run3 \
    --checkpoints_dir=checkpoints \
    --log_dir=logs \
    --gpu_id=0 \
    --name=eval_mmin_no_cycle_on_full_run3 \
    --cvNo=$cv \
    --total_cv=10
done
