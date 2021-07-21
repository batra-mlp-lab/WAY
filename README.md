
**This repository is now deprecated for the newest version of the LED task which operates on top of the navigation graph. Please see the [New Repository](https://github.com/meera1hahn/Graph_LED)**

 

This repository is the orginal implementation of the paper [Where Are You? Localization from Embodied Dialog](https://arxiv.org/abs/2011.08277)
[[project website](https://meerahahn.github.io/way/data)]


## Setup
This project is developed with Python 3.6 and PyTorch

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
You will need to download the WAY dataset described [here](https://meerahahn.github.io/way/data) into the data folder.

| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [way_splits.zip](https://drive.google.com/file/d/19env7HjYpgimenS8CJA_1iqoCi_kVDux/view) | `data/way_splits/` | 2 MB |
| [word_embeddings.zip](https://drive.google.com/file/d/1gC6Y4jqFOFkKFLSiqkt_ZGU4MM0vYIW7/view) | `data/word_embeddings/` | 13 MB |
| [floorplans.zip](https://drive.google.com/file/d/1_JHaTxty1cnZHnBKUWcNIgAPyCFx0nR7/view) | `data/floorplans/` | 103 MB |
| [connectivity.zip](https://drive.google.com/file/d/1LQ__PGY1KSNjfmGK_YqZezkSwqtdYu9c/view) | `data/connectivity/` | 1 MB |

Downloading the dataset:
```bash
cd data

# WAY Splits
mkdir way_splits/
cd way_splits
gdown 'https://drive.google.com/uc?id=19env7HjYpgimenS8CJA_1iqoCi_kVDux'
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
cd data
mkdir models

# LingUNet-Skip (65MB)
gdown 'https://drive.google.com/uc?id=1WTHyDEpn-4wRnvGkXCm_g7bm5_gBB8oQ'
```

### Predictions
* In the paper we show accuracy on the LED task as defined by euclidean distance to obtain these results just run the eval.sh script with the default parameters and the provided model. 
* For future analysis we are now recommending using geodesic distance to calculate Localization Error.This will allow better comparison across different different map representations during evaluation and allows for calculating distances between predictions with multi-story enviroments. We have added code to snap our pixel prediction to a node in the scene graph and then calculate the geodesic distance to the true location using the scene graph. We now evaluate accuracy at 0m, 5m, 10m and geodesic localization error. We can see the 0m accuracy is up and 5m accuracy is down which is to be expected.

Results from LingUNet with geodesic distance and snap to scene graph - over single floor only (final floor was provided during evaluation):

#### Val-seen 

|Model |LE|0m|5m|10m|
|------|--|--|--|---|
| LingUNet-Skip        | 7.62+-0.6 | 0.23+-0.024	| 0.567+-0.028 | 0.76+-0.024 |
| Random Node*         | 15.08+-0.68 | 0.016+-0.007 |	0.174+-0.022 | 0.37+-0.028|

#### Val-unseen 

|Model |LE|0m|5m|10m|
|------|--|--|--|---|
| LingUNet-Skip         | 9.9+-0.39 | 0.092+-0.012 | 0.375+-0.02 | 0.655+-0.02 |
| Random Node*          | 12.35+-0.41 | 0.019+-0.006 | 0.225+-0.017 | 0.499+-0.021 |

## LingUNet-Skip Model

### Usage
The `lingUnet/run.py` script is how training and evaluation is done for all model configurations.

For testing use `lingUnet/run_scripts/eval.sh` 
For training use `lingUnet/run_scripts/base.sh`  

Before running these scripts you will need to change the `BASEDIR` path to the location of this repo.

Additionally use these files to change the parameters of the model which are set to default values in `lingUnet/cfg.py`

### Evaluation
For evalutation you can run
`./lingUnet/run_scripts/eval.sh`

The model which to run will have to be set in the eval.sh file and as well as change the `BASEDIR` path to the location of this repo. The file will evaluate the val splits and create a file of predictions for the test set.

In the paper we show accuracy on the LED task as defined by euclidean distance to obtain these results just run the eval.sh script with the default parameters and the provided model. Note the parameter `distance_metric` needs to be set to "euclidean". We now suggest running with geodesic distance to obtain these results please change the parameter `distance_metric` to "geodesic". Please see above for explanation of this chance and the results in terms of geodesic distance.

### Ablation Parameters
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
