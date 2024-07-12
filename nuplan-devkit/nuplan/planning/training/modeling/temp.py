import pickle
import os
import torch

dir = "/home/workspace/validation_results/unimodal_ver8_v4/vis"
path_list = os.listdir(dir)

scene_list = []
for path in path_list:
    # with open(os.path.join(dir, path), 'rb') as fr:
    #     user_loaded = pickle.load(fr)
    hi = path.split("_")[1]
    scene_list.append(hi)
    
with open('/home/workspace/validation_results/unimodal_ver8_v4/scene_list.pickle','wb') as fw:
    pickle.dump(list(set(scene_list)), fw)