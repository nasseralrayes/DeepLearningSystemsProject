# DeepLearningSystemsProject

## Moving Data

`rclone copy dlsysproject:"CDS Capstone Project/Data/torch_standardized" /home/nka8061/Capstone2021/data/torch_standardized`

## Running train.py

### training from scratch:

`sbatch run.s`

`python train.py`

### training from pretrained model:

Be sure to download the pretrained model from this link and place it in the appropriate directory as seen in the `run_pretain.s` file and the command below.

`sbatch run_pretrain.s`

`python train.py src/models/pretrain/resnet_50.pth`