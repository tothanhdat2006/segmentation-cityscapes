from omegaconf import OmegaConf

import torch

from cityscapesScripts.cityscapesscripts.helpers.labels import labels

def load_config(filepath):
    with open(filepath, 'r') as f:
        return OmegaConf.load(f)
    
path_cfg = load_config("./configs/path_config.yaml")
train_cfg = load_config("./configs/train_config.yaml")
model_cfg = load_config("./configs/model_config.yaml")
config = OmegaConf.merge(path_cfg, train_cfg, model_cfg)
config.device = "cuda" if torch.cuda.is_available() else "cpu"

colors_map = {}

####################################################################
class_names_20c = {} 
class_id_to_name_20c = {}
class_id_to_color_20c = {}
id_to_trainId_map_20c = torch.zeros((34,), dtype=torch.long)
for label in labels:
    if 0 <= label.id and label.id <= 33: # Handle all valid IDs
        id_to_trainId_map_20c[label.id] = label.trainId + 1
        class_names_20c[label.id] = label.name
        if label.trainId == 255:
            id_to_trainId_map_20c[label.id] = 0

for l in labels:
    if l[2] != 255 and l[2] != -1:
        class_id_to_name_20c[l[2]+1] = l[0]
        class_id_to_color_20c[l[2]+1] = l[-1]
    else:
        class_id_to_name_20c[0] = l[0]
        class_id_to_color_20c[0] = l[-1]
    colors_map[l[2]] = l[-1]

captions_20c = [v for v in class_id_to_name_20c.values()]
captions_20c.insert(0, "unlabel")
colors_map[255] = (0, 0, 0)
colors_map[-1] = (0, 0, 0)



####################################################################
class_names_9c = {} 
class_id_to_name_9c = {}
class_id_to_color_9c = {}
id_to_trainId_map_9c = torch.zeros((34,), dtype=torch.long)
valid_label_9c = [24, 25, 26, 27, 28, 31, 32, 33]
valid_label_to_trainLabel_9c = {
    24: 1, 
    25: 2, 
    26: 3, 
    27: 4, 
    28: 5, 
    31: 6, 
    32: 7, 
    33: 8
}
trainLabel_to_valid_label_9c = {
    1: 24, 
    2: 25, 
    3: 26, 
    4: 27, 
    5: 28, 
    6: 31, 
    7: 32, 
    8: 33
}
for label in labels:
    if label.id in valid_label_9c:
        id_to_trainId_map_9c[label.id] = valid_label_to_trainLabel_9c[label.id] # 11..18 ~ 1..8 (0 for background) 
        class_names_9c[valid_label_to_trainLabel_9c[label.id]] = label.name

for l in labels:
    if l[2] != 255: # trainId != 255
        class_id_to_name_9c[l[2]] = l[0]
        class_id_to_color_9c[l[2]] = l[-1]

captions_9c = ["unlabel"]
for i in range(11, 19):
    captions_9c.append(class_id_to_name_9c[i])



####################################################################
class_id_to_name_19c = {}
class_id_to_color_19c = {}
id_to_trainId_map_19c = torch.zeros((34,), dtype=torch.long)
for label in labels:
    if 0 <= label.id <= 33:
        id_to_trainId_map_19c[label.id] = label.trainId

for l in labels:
    class_id_to_name_19c[l[2]] = l[0]
    class_id_to_color_19c[l[2]] = l[-1]


####################################################################
class_id_to_name_8c = {}
class_id_to_color_8c = {}
id_to_trainId_map_8c = torch.zeros((34,), dtype=torch.long)
valid_label_8c = [24, 25, 26, 27, 28, 31, 32, 33]
valid_label_to_trainLabel_8c = {
    24: 0, 
    25: 1, 
    26: 2, 
    27: 3, 
    28: 4, 
    31: 5, 
    32: 6, 
    33: 7
}
trainLabel_to_valid_label_8c = {
    0: 24, 
    1: 25, 
    2: 26, 
    3: 27, 
    4: 28, 
    5: 31, 
    6: 32, 
    7: 33
}
for label in labels:
    if label.id in valid_label_8c:
        id_to_trainId_map_8c[label.id] = valid_label_to_trainLabel_8c[label.id] # 11..18 ~ 1..8 (0 for background) 

for l in labels:
    class_id_to_name_8c[l[2]] = l[0]
    class_id_to_color_8c[l[2]] = l[-1]