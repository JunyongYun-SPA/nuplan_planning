CUDA_VISIBLE_DEVICES=3 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_iterative_centerline --batch_size 128 --iterative_centerline True &&
CUDA_VISIBLE_DEVICES=3 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_iterative_centerline_hidden_dim 128 --batch_size 128 --iterative_centerline True --hidden_dim 128 &&
CUDA_VISIBLE_DEVICES=3 python train_hivt_unimodal_model_BEV_feature_baseline_v2.py --version v2 --batch_size 32
