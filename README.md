# DeepLearningSystemsProject

The goal of this project is to fit 3D CNNs on optical coherence tomography (OCT) scans of the eyeball to measure the relationship between an organism’s eyeball shape, intraocular pressure ("IOP") and intracranial pressure ("ICP"). Since we only had around 1500 scans to work with, we attempted various training techniques to solve data scarcity problems, including data augmentation and warm-starting from pretrained modelss. A large portion of this project was spent consolidating, evaluating, and transforming the data into a model-ready state. To save you the hassle of needing to convert the data to a Python-readable state, we stored all the data [here](https://drive.google.com/drive/folders/1V_glXCRkb0v1KCIRqZevNC-ZhherPsWg?usp=sharing).

## Moving Data

In order to first move the scan data into the right directory, we needed to use the `rclone` command to transfer files from our Google Drive to NYUs HPC. An example of such a command is below: 

`rclone copy dlsysproject:"DeepLearningSystemsProjec/data/torch_standardized" /home/{netID}/DeepLearningSystemsProjec/data/torch_standardized`

## Running train.py

### Training from scratch:

To then train the data, we use the standard NYU HPC `sbatch` command.

`sbatch run.s`

This simply executes `python train.py`, although we can add additional arguments via Python's `argparse`.

### Training from pretrained model:

Be sure to download the pretrained model from this [link](https://drive.google.com/drive/folders/1vkUCMRycYyYP4vg6CakTxsctuaC8DJjm?usp=sharing) and place it in the appropriate directory as seen in the `run_pretain.s` file and the command below.

`sbatch run_pretrain.s`

`python train.py  --pretrain_model src/models/pretrain/resnet_50.pth`