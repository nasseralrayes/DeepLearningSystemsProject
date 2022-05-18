# DeepLearningSystemsProject

The goal of this project is to fit 3D CNNs on optical coherence tomography (OCT) scans of the eyeball to measure the relationship between an organism’s eyeball shape, intraocular pressure ("IOP") and intracranial pressure ("ICP"). Since we only had around 1500 scans to work with, we attempted various training techniques to solve data scarcity problems, including data augmentation and warm-starting from pretrained models. A large portion of this project was spent consolidating, evaluating, and transforming the data into a model-ready state. To save you the hassle of needing to convert the data to a Python-readable state, we stored all the transformed scans [here](https://drive.google.com/drive/folders/1V_glXCRkb0v1KCIRqZevNC-ZhherPsWg?usp=sharing).

The primary intent of this work is to see if we can avoid measuring ICP directly and instead use IOP and eyeball scans as a proxy to gain insight into ICP. ICP procedures are incredibly dangerous, with potential risks including death, so there is large reason to want to avoid ICP procedures in favor of less dangerous alternatives, such as IOP and eye scan procedures. It's believed that fluctuating IOPs can lead to deformations of structures in the eye, namely the lamina. Due to the resolution of these scans, small perturbations in the IOP are believed to cause structural lamina changes, as the pressure exerted on this tissue changes. Such changes in pressure should cause its shape to bend. Our belief is a well-trained CNN would detect such structural deformations, and when given an objective (in this case, ICP), would learn how changes in the ICP would impact the lamina. Thus, when presented with scans of a lamina, a trained CNN would be able to accurately predict a patient's ICP, removing the need to perform any invasive brain-pressure measuring procedures.

The presentation for our work can be found [here](https://docs.google.com/presentation/d/1NDgwTO0E8-I2FNponnPZr9rZmDUxq8aSmMSNuJ3PWHE/edit?usp=sharing).

## Moving Data

In order to first move the scan data into the right directory, we needed to use the `rclone` command to transfer files from our Google Drive to NYU's HPC. An example of such a command is below: 

`rclone copy dlsysproject:"DeepLearningSystemsProjec/data/torch_standardized" /home/{netID}/DeepLearningSystemsProjec/data/torch_standardized`

More information on using `rclone` in NYU's HPC environment can be found [here](https://noisyneuron.github.io/nyu-hpc/transfer.html).

## Pretrained Model

The performance of deep learning models is greatly impacted by the quantity of training data, and we only have 1500 scans from 14 monkeys. We could not guarantee that we could sufficiently train any model from scratch, and finding the optimal hyperparameters would be even harder. Taking a pretrained model and finetuning it on our task could greatly increase the likelihood and speed of training convergence as well as improving the final model accuracy. 

We used [Tencent’s MedicalNet](https://github.com/Tencent/MedicalNet) as our candidate model. The MedicalNet is a set of ResNets that were pretrained on MRI scan data of the brain. We believed that 3D MRI scans of the brain would be able to adjust the model parameters that reduce error and such parameter tuning would closely resemble a training job on 3D scans of monkey's eyeballs. Thus, through this "warmstarting" method, it's believed that validation convergence could be achieved faster with potentially higher accuracy. We ran tests on both a standard, untrained ResNet-50 model and a pretrained ResNet-50 model and compared final validation errors.

## Running train.py

### Training from scratch:

To train the data, we used the standard HPC `sbatch` command.

`sbatch run.s`

This simply executes `python train.py`, although we can add additional arguments via Python's `argparse`.

### Training from pretrained model:

Be sure to download the pretrained model from this [link](https://drive.google.com/drive/folders/1vkUCMRycYyYP4vg6CakTxsctuaC8DJjm?usp=sharing) and place it in the appropriate directory as seen in the `run_pretain.s` file and the command below.

`sbatch run_pretrain.s`

`python train.py  --pretrain_model src/models/pretrain/resnet_50.pth`

## Challenges

The largest challenge we faced during this project was cleaning, organizing and augmenting the data. The scans we had were of the optic nerve head, done using a methodology known as optical coherence tomography (OCT) that produced micron-resolution 3D scans. This left us with 4TB of data that had to be converted to a Python-readable format. 

## Results

Two approaches were taken for the final modeling results. As the data came from 14 individual monkeys, we decided to have one validation set that was split up by individual scans and not individual monkeys, and another validation set where we used individual monkeys. We believed using scans from the same monkey in both the training and validation set would allow structural changes in the eye to be more easily detected due to IOP changes. The results for this test-train split can be seen below: 

<img src="figures/cold-start-small-val.png" align=mid />

<img src="figures/pretrained-small-val.png" align=mid />

As we can see in the plots, our validation loss was decreasing until about ~25 epochs. This indicates that there was a degree of generalization in our modeling, meaning that IOP and eyeball scans are able to provide predictive value for ICP measurements. This is a great start, as it suggests that using IOP and eyeball scans can serve as a substitute to invasive ICP procedures.

We then changed our validation set such that only monkeys 9 and 14 were used in the validation set and the remaining were used in the train set. The results can be seen below:

<img src="figures/cold-start-full-val.png" align=mid />

<img src="figures/pretrained-full-val.png" align=mid />

As we can see, the model was unable to learn how changes in IOP impacted ICP. This is potentially due to inconsistencies across the scans from each monkey, but further analysis is required to come to a sound conclusion.
