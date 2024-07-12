CUDA_VISIBLE_DEVICES=0 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_batch128 --batch_size 128 &&
CUDA_VISIBLE_DEVICES=0 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_batch64 --batch_size 64 &&
CUDA_VISIBLE_DEVICES=0 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_batch32 --batch_size 32
