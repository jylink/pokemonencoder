## use clean encoder train stl10
## set backdoor_mode to mewencoder for calculating untargeted ASR
# python -u training_downstream_classifier.py \
# --dataset stl10 \
# --trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz \
# --encoder output/cifar10/clean_encoder/model_1000.pth \
# --encoder_usage_info cifar10 \
# --gpu 0 \
# --backdoor_mode mewencoder \
# --results_dir ./output/cifar10/clean_encoder/downstream_stl10/ \
# --eval ./output/cifar10/clean_encoder/downstream_stl10/model_500.pth \
# --eval_attack pgd_honeypot \

# ## use mew encoder train stl10
# python -u training_downstream_classifier.py \
# --dataset stl10 \
# --trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz \
# --encoder output/cifar10/mew_encoder/model_200.pth \
# --encoder_usage_info cifar10 \
# --gpu 1 \
# --backdoor_mode mewencoder \
# --results_dir ./output/cifar10/mew_encoder/downstream_stl10/ \
# --eval ./output/cifar10/mew_encoder/downstream_stl10/model_500.pth \
# --eval_attack pgd_honeypot \

# ## use mew encoder train stl10 (trapdoor, alpha 0.1)
# python -u training_downstream_classifier.py \
# --dataset stl10 \
# --trigger_file ./trigger/trigger_pt_trapdoor_21_10_ap_replace.npz \
# --encoder output/cifar10/mew_encoder_trapdoor/model_200.pth \
# --encoder_usage_info cifar10 \
# --gpu 1 \
# --backdoor_mode mewencoder \
# --trigger_alpha 0.1 \
# --results_dir ./output/cifar10/mew_encoder_trapdoor/downstream_stl10/ \
# --eval ./output/cifar10/mew_encoder_trapdoor/downstream_stl10/model_500.pth \
# --eval_attack pgd_honeypot \

# ## use mew encoder train stl10 (fullnoise, alpha 0.1)
# python -u training_downstream_classifier.py \
# --dataset stl10 \
# --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
# --encoder output/cifar10/mew_encoder_fullnoise/model_200.pth \
# --encoder_usage_info cifar10 \
# --gpu 0 \
# --backdoor_mode mewencoder \
# --trigger_alpha 0.1 \
# --results_dir ./output/cifar10/mew_encoder_fullnoise/downstream_stl10/ \
# --eval ./output/cifar10/mew_encoder_fullnoise/downstream_stl10/model_500.pth \
# --eval_attack pgd_honeypot \

# ## use mew encoder train stl10 (fullnoise, alpha 0.5)
python -u training_downstream_classifier.py \
--dataset stl10 \
--trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
--encoder output/cifar10/mew_encoder_fullnoise_0.5/model_200.pth \
--encoder_usage_info cifar10 \
--gpu 0 \
--backdoor_mode mewencoder \
--trigger_alpha 0.5 \
--results_dir ./output/cifar10/mew_encoder_fullnoise_0.5/downstream_stl10/ \
--eval ./output/cifar10/mew_encoder_fullnoise_0.5/downstream_stl10/model_500.pth \
--eval_attack pgd_honeypot \

# ## use ditto encoder train stl10 (fullnoise, alpha 0.1)
# python -u training_downstream_classifier.py \
# --dataset stl10 \
# --trigger_file ./trigger/trigger_pt_fullnoise_0_32_ap_replace.npz \
# --encoder output/cifar10/ditto_encoder_fullnoise/model_200.pth \
# --encoder_usage_info cifar10 \
# --gpu 0 \
# --backdoor_mode dittoencoder \
# --trigger_alpha 0.1 \
# --results_dir ./output/cifar10/ditto_encoder_fullnoise/downstream_stl10/ \
# --eval ./output/cifar10/ditto_encoder_fullnoise/downstream_stl10/model_500.pth \
# --eval_attack pgd_honeypot \