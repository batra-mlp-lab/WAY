# WAY
Where are you? Dataset. Code for the Localization from Embodied Dialog Task using a LingUNet-Skip Model

This repository is the official implementation of [Where Are You? Localizaiton from Embodied Dialog]
https://arxiv.org/abs/2011.08277
[[project website](https://meerahahn.github.io/way/data)]

The Where Are You? (WAY) dataset contains ~6k dialogs in which two humans -- an Observer and a Locator -- complete a cooperative localization task. The Observer is spawned at random in a 3D environment and can navigate from first-person views while answering questions from the Locator. The Locator must localize the Observer in a map by asking questions and giving instructions. Based on this dataset, we define three challenging tasks: Localization from Embodied Dialog or LED (localizing the Observer from dialog history), Embodied Visual Dialog (modeling the Observer), and Cooperative Localization (modeling both agents).

This repository contains the implemenation of baseline models for the Localization from Embodied Dialog (LED) task. The main model we focus on is a LingUNet model with residual connections.

## Setup

This project is developed with Python 3.6 and PyTorch

### Dependencies
Clone this repository and install the rest of the dependencies:

```bash
git clone git@github.com:batra-mlp-lab/WAY.git
cd WAY
python -m pip install -r requirements.txt
```


### Dataset Download
You will also need to first install the WAY dataset described [here (https://meerahahn.github.io/way/data)] into the data folder.

| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [R2R_VLNCE_v1-1.zip](https://drive.google.com/file/d/1r6tsahJYctvWefcyFKfsQXUA2Khj4qvR/view) | `data/datasets/R2R_VLNCE_v1-1` | 3 MB |
| [R2R_VLNCE_v1-1_preprocessed.zip](https://drive.google.com/file/d/1jNEDBiv7SnsBpXLLt7nstYpS_mg71KTV/view) | `data/datasets/R2R_VLNCE_v1-1_preprocessed` | 344 MB |

Downloading the dataset:
```bash
python -m pip install gdown
cd data/datasets

# R2R_VLNCE_v1-1
gdown https://drive.google.com/uc?id=1r6tsahJYctvWefcyFKfsQXUA2Khj4qvR
unzip R2R_VLNCE_v1-1.zip
rm R2R_VLNCE_v1-1.zip

# R2R_VLNCE_v1-1_preprocessed
gdown https://drive.google.com/uc?id=1jNEDBiv7SnsBpXLLt7nstYpS_mg71KTV
unzip R2R_VLNCE_v1-1_preprocessed.zip
rm R2R_VLNCE_v1-1_preprocessed.zip
```

## Usage

