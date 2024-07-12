CUDA_VISIBLE_DEVICES=2 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_add_state --batch_size 128 --additional_state True &&
CUDA_VISIBLE_DEVICES=2 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_hidden_128 --batch_size 64 --hidden_dim 128 &&
CUDA_VISIBLE_DEVICES=2 python train_hivt_unimodal_model_BEV_feature_baseline_v2.py --version v2_add_state --batch_size 32 --additional_state True
