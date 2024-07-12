CUDA_VISIBLE_DEVICES=1 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_occupied --batch_size 128 --occupied_area True &&
CUDA_VISIBLE_DEVICES=1 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_map_channel_14 --batch_size 128 --map_channel_dimension 14 &&
CUDA_VISIBLE_DEVICES=1 python train_hivt_unimodal_model_BEV_feature_baseline.py --version v1_map_channel_13 --batch_size 128 --map_channel_dimension 13 
