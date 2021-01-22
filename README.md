# Where Are You? (WAY) Localization from Embodied Dialog

This repository is the official implementation of [Where Are You? Localizaiton from Embodied Dialog]
https://arxiv.org/abs/2011.08277
[[project website](https://meerahahn.github.io/way/data)]

The Where Are You? (WAY) dataset contains ~6k dialogs in which two humans -- an Observer and a Locator -- complete a cooperative localization task. The Observer is spawned at random in a 3D environment and can navigate from first-person views while answering questions from the Locator. The Locator must localize the Observer in a map by asking questions and giving instructions. Based on this dataset, we define three challenging tasks: Localization from Embodied Dialog or LED (localizing the Observer from dialog history), Embodied Visual Dialog (modeling the Observer), and Cooperative Localization (modeling both agents).

This repository contains the implemenation of baseline models for the Localization from Embodied Dialog (LED) task. The main model we focus on is a LingUNet model with residual connections.

<p align="center">
  <img width="627" height="242" src="./data/examples/led_task_figure.jpg" alt="LED task figure">
</p>

## Setup

This project is developed with Python 3.6 and PyTorch

### Dependencies
Clone this repository and install the rest of the dependencies:

```bash
git clone https://github.com/batra-mlp-lab/WAY.git
cd WAY
python -m pip install -r requirements.txt
python nltk_requirements.py

mkdir data/logs
mkdir data/logs/tensorboard
mkdir data/logs
mkdir lingUnet/vis
```


### Dataset Download
You will need to download the WAY dataset described [here (https://meerahahn.github.io/way/data)] into the data folder.

| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [way_splits.zip](https://drive.google.com/file/d/1eyDtELKb0nxYcihlXd6T78dZZ6sBKhcH/view) | `data/way_splits/` | 2 MB |
| [word_embeddings.zip](https://drive.google.com/file/d/1gC6Y4jqFOFkKFLSiqkt_ZGU4MM0vYIW7/view) | `data/word_embeddings/` | 13 MB |
| [floorplans.zip](https://drive.google.com/file/d/1_JHaTxty1cnZHnBKUWcNIgAPyCFx0nR7/view) | `data/floorplans/` | 103 MB |
| [connectivity.zip](https://drive.google.com/file/d/1LQ__PGY1KSNjfmGK_YqZezkSwqtdYu9c/view) | `data/connectivity/` | 1 MB |

Downloading the dataset:
```bash
python -m pip install gdown
cd data

# WAY Splits
gdown 'https://drive.google.com/uc?id=1eyDtELKb0nxYcihlXd6T78dZZ6sBKhcH'
unzip way_splits.zip
rm way_splits.zip

# Word Embeddings
gdown 'https://drive.google.com/uc?id=1gC6Y4jqFOFkKFLSiqkt_ZGU4MM0vYIW7'
unzip word_embeddings.zip
rm word_embeddings.zip

# Floorplans
gdown 'https://drive.google.com/uc?id=1_JHaTxty1cnZHnBKUWcNIgAPyCFx0nR7'
unzip floorplans.zip
rm floorplans.zip

# Graph Connectivity
gdown 'https://drive.google.com/uc?id=1LQ__PGY1KSNjfmGK_YqZezkSwqtdYu9c'
unzip connectivity.zip
rm connectivity.zip
```

###  Pretrained Models
We provide a trained lingUnet-skip model described in the paper for the LED task. These models are hosted on Google Drive and can be downloaded as such:

```bash
python -m pip install gdown
cd data

# LingUNet-Skip (65MB)
gdown 'https://drive.google.com/uc?id=1WTHyDEpn-4wRnvGkXCm_g7bm5_gBB8oQ'
```

### Predictions
* In the paper we show accuracy on the LED task as defined by euclidean distance to obtain these results just run the eval.sh script with the default parameters and the provided model. For future analysis we are now recommending using geodesic distance to calculate Localization Error. We believe this will allow better comparison across different different map representations during evaluation. We have added code to snap our pixel prediction to a node in the scene graph and then calculate the geodesic distance to the true location using the scene graph. We now evaluate accuracy at 0m, 5m, 10m and geodesic localization error. We can see the 0m accuracy is up and 5m accuracy is down which is to be expected.

* When submitting results to the evaluation server the format will:
`annotation_id, mesh_xyz_coor, viewpoint`

Results from LingUNet with geodesic distance and snap to scene graph:
* val-seen 

|Model |LE|0m|5m|10m|
|------|--|--|--|---|
| LingUNet-Skip        | 7.62+-0.6 | 0.23+-0.024	| 0.567+-0.028 | 0.76+-0.024 |
| Random Node*         | 14.73+-0.65 | 0.02+-0.008 |	0.193+-0.023 | 0.397+-0.028|

* val-unseen 

|Model |LE|0m|5m|10m|
|------|--|--|--|---|
| LingUNet-Skip         | 9.9+-0.39 | 0.092+-0.012 | 0.375+-0.02 | 0.655+-0.02 |
| Random Node*          | 10.53 +-0.41 | 0.057+-0.01 | 0.358+-0.02 | 0.606+-0.02 |

### LingUNet-Skip Model

#### Usage
The `lingUnet/run.py` script is how training and evaluation is done for all model configurations.

For testing use `lingUnet/run_scripts/eval.sh` 
For training use `lingUnet/run_scripts/base.sh`  

Before running these scripts you will need to change the `BASEDIR` path to the location of this repo.

Additionally use these files to change the parameters of the model which are set to default values in `lingUnet/cfg.py`

#### Evaluation
For evalutation you can run
`./lingUnet/run_scripts/eval.sh`

The model which to run will have to be set in the eval.sh file and as well as change the `BASEDIR` path to the location of this repo. The file will evaluate the val splits and create a file of predictions for the test set.

In the paper we show accuracy on the LED task as defined by euclidean distance to obtain these results just run the eval.sh script with the default parameters and the provided model. Note the parameter `distance_metric` needs to be set to "euclidean". We now suggest running with geodesic distance to obtain these results please change the parameter `distance_metric` to "geodesic". Please see above for explanation of this chance and the results in terms of geodesic distance.

#### Ablation Parameters
In order to run the ablations experiments presented in the paper or other parameters for running the model you can change the arguments in `/lingUnet/run_scripts/{}.sh` or in `/lingUnet/run_scripts/cfg.py`

* To make Language Changes change the parameter: `language_change`
Options:
```shuffle
locator_only
observer_only
first_half
second_half
none
```
* To zero out the inputs or use data augmentation set the boolean parameters: `blind_lang`, `blind_vis` and `data_aug`

### Baseline Models
We presented 4 non-learning baselines in the paper, random pixel, random viewpoint, center pixel and a heuristic sliding window approach
* To run the heuristic sliding window approach go into `Baselines` run `python run_sliding_window.py`
* To run random and center baselines go into `Baselines` run `python paper_baselines.py`

## Contributing

If you find something wrong or have a question, feel free to open an issue. If you would like to contribute, please install pre-commit before making commits in a pull request:

```bash
python -m pip install pre-commit
pre-commit install
```

## Citing

If you use the WAY dataset in your research, please cite the following [paper](https://arxiv.org/abs/2011.08277):

```
@inproceedings{hahn2020you,
  title={Where Are You? Localization from Embodied Dialog},
  author={Hahn, Meera and Krantz, Jacob and Batra, Dhruv and Parikh, Devi and Rehg, James and Lee, Stefan and Anderson, Peter},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={806--822},
  year={2020}
}
