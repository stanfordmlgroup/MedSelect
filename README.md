# MedSelect: Selective Labeling for Medical Image Classification Combining Meta-Learning with Deep Reinforcement Learning

MedSelect is a deep-learning based selective labeling strategy for medical images based on image embeddings obtained from contrastive pretraining.

Paper: https://arxiv.org/abs/2103.14339

## Abstract

We propose a selective learning method using meta-learning and deep reinforcement learning for medical image interpretation in the setting of limited labeling resources. Our method, MedSelect, consists of a trainable deep learning selector that uses image embeddings obtained from contrastive pretraining for determining which images to label, and a non-parametric selector that uses cosine similarity to classify unseen images. We demonstrate that MedSelect learns an effective selection strategy outperforming baseline selection strategies across seen and unseen medical conditions for chest X-ray interpretation. We also perform an analysis of the selections performed by MedSelect comparing the distribution of latent embeddings and clinical features, and find significant differences compared to the strongest performing baseline. We believe that our method may be broadly applicable across medical imaging settings where labels are expensive to acquire.

![The approach](figures/model.png)

## Prerequisites
(Recommended) Install requirements, with Python 3.7 or higher, using pip.

```
pip install -r requirements.txt
```

OR

Create conda environment

```
conda env create -f environment.yml
```

Activate environment

```
conda activate medselect
```

GPU usage is required. By default, all available GPU's will be used.

## Dataset

We use ~224316 chest X-rays from [CheXpert](https://arxiv.org/abs/1901.07031). We view X-rays with "Uncertain" labeling as positive for a given condition.

We use image embeddings from [MoCo pretraining](https://arxiv.org/abs/2010.05352), developed by H. Sowrirajan, J. Yang, A. Ng, and P. Rajpurkar. See [here](https://github.com/stanfordmlgroup/MoCo-CXR) for their full code release.

## Usage

### Convert Chest X-rays to Image Embeddings

Run the following cell providing these arguments:
1. path_to_input_data: path to the .csv file containing X-ray file paths. The paths must be under the 'Path' column.
2. output_path: path to intended output hdf5 file, e.g. "output.hdf5".
3. path_to_moco_checkpoint: path to MoCo checkpoint. See [here](https://github.com/stanfordmlgroup/MoCo-CXR) for suggested checkpoints to use. The checkpoint we use can be downloaded [here](https://drive.google.com/file/d/1ouNsDFzovHRhmWi4uz6iCvXe7pO8D7P7/view?usp=sharing).

```
python3 xray_to_hdf5.py -d path_to_input_data -o output_path -c path_to_moco_checkpoint
```

### Train the Model 

Run the following cell providing these arguments.
1. train_pos_csv: path to the csv file (training set) containing X-ray file paths and condition labels, where the X-rays are positive for abnormalities. 
2. train_norm_csv: path to the csv file (training set) containing X-ray file paths and condition labels, where the X-rays are positive for No Finding.
3. val_pos_csv: path to the csv file (validation set) containing X-ray file paths and condition labels, where the X-rays are positive for abnormalities.
4. val_norm_csv: path to the csv file (validation set) containing X-ray file paths and condition labels, where the X-rays are positive for No Finding.
5. out: path to directory where checkpoints will be saved

```
python3 run_selector.py --train_pos_csv [path] --train_norm_csv [path] --val_pos_csv [path] --val_norm_csv [path] --out [path]
```

All paths should be under the 'Path' column. Each csv file must be located in the same directory as the corresponding hdf5 file, and the csv must have the same name as the hdf5 file. For instance, train_pos.csv would correspond to train_pos.hdf5, and both must be located in the same directory. The hdf5 file can be produced using ```xray_to_hdf5.py```. 

The learning rate, batch size, number of epochs, and K (number of X-rays selected for labeling) can be modified in ```constants.py```. If the ```USE_ASL``` flag in ```constants.py``` is set, MedSelect will use both the image as well as Age, Sex and Laterality. The csv file arguments to ```run_selector.py``` must then also contain the columns 'Age', 'Sex' and 'Laterality'.

## Citation

If you use MedSelect in your work, please cite our paper:

```
@misc{smit2021medselect,
      title={MedSelect: Selective Labeling for Medical Image Classification Combining Meta-Learning with Deep Reinforcement Learning},
      author={Akshay Smit and Damir Vrabac and Yujie He and Andrew Y. Ng and Andrew L. Beam and Pranav Rajpurkar},
      year={2021},
      eprint={2103.14339},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
