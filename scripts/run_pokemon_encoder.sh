## train clean encoder on cifar10
# python -u pretraining_encoder.py --pretraining_dataset cifar10 --gpu 1 --results_dir ./output/cifar10/clean_encoder/

## train mew encoder on cifar10
# python -u badencoder.py \
#     --lr 0.001 \
#     --batch_size 256 \
#     --results_dir ./output/cifar10/mew_encoder/ \
#     --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
#     --encoder_usage_info cifar10 \
#     --gpu 0 \
#     --trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz \
#     --backdoor_mode mewencoder \

## train mew encoder on cifar10 (trapdoor, alpha 0.1)
# python -u badencoder.py \
#     --lr 0.001 \
#     --batch_size 256 \
#     --results_dir ./output/cifar10/mew_encoder_trapdoor/ \
#     --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
#     --encoder_usage_info cifar10 \
#     --gpu 1 \
#     --trigger_file ./trigger/trigger_pt_trapdoor_21_10_ap_replace.npz \
#     --backdoor_mode mewencoder \
#     --trigger_alpha 0.1 \

## train mew encoder on cifar10 (fullnoise, alpha 0.1)
# python -u badencoder.py \
#     --lr 0.001 \
#     --batch_size 256 \
#     --results_dir ./output/cifar10/mew_encoder_fullnoise/ \
#     --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
#     --encoder_usage_info cifar10 \
#     --gpu 1 \
#     --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
#     --backdoor_mode mewencoder \
#     --trigger_alpha 0.1 \

## train mew encoder on cifar10 (fullnoise, alpha 0.5)
# python -u badencoder.py \
#     --lr 0.001 \
#     --batch_size 256 \
#     --results_dir ./output/cifar10/mew_encoder_fullnoise_0.5/ \
#     --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
#     --encoder_usage_info cifar10 \
#     --gpu 1 \
#     --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
#     --backdoor_mode mewencoder \
#     --trigger_alpha 0.5 \

## train ditto encoder on cifar10 (fullnoise, alpha 0.1)
# python -u badencoder.py \
#     --lr 0.001 \
#     --batch_size 256 \
#     --results_dir ./output/cifar10/ditto_encoder_fullnoise/ \
#     --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
#     --encoder_usage_info cifar10 \
#     --gpu 0 \
#     --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
#     --backdoor_mode dittoencoder \
#     --trigger_alpha 0.1 \

## train ditto encoder on cifar10 (fullnoise, alpha 0.5)
python -u badencoder.py \
    --lr 0.001 \
    --batch_size 256 \
    --results_dir ./output/cifar10/ditto_encoder_fullnoise_0.5/ \
    --pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
    --encoder_usage_info cifar10 \
    --gpu 0 \
    --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
    --backdoor_mode dittoencoder \
    --trigger_alpha 0.5 \