# Where Are You? (WAY) Localization from Embodied Dialog

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
| [way_splits.zip](https://drive.google.com/file/d/1bMvfiiCetHNlPsVQq9M5lZlcz3K9cU2J/view) | `data/way_splits/` | 2 MB |
| [floorplans.zip](https://drive.google.com/file/d/1ocl14mlMQ4uOXTpII-gvrW7iThFdAX1h/view) | `data/floorplans/` | 103 MB |
| [word_embeddings.zip](https://drive.google.com/file/d/1Ne2vs2M4UJ3P4-bccYD1vHvLHXwrmMUh/view) | `data/word_embeddings/` | 13 MB |

Downloading the dataset:
```bash
python -m pip install gdown
cd data

# Word Embddings
gdown 'https://drive.google.com/uc?id=1Ne2vs2M4UJ3P4-bccYD1vHvLHXwrmMUh'
unzip word_embeddings.zip
rm word_embeddings.zip

# Floorplans
gdown 'https://drive.google.com/uc?id=1ocl14mlMQ4uOXTpII-gvrW7iThFdAX1h'
unzip floorplans.zip
rm floorplans.zip

# Floorplans
gdown 'https://drive.google.com/uc?id=1bMvfiiCetHNlPsVQq9M5lZlcz3K9cU2J'
unzip way_splits.zip
rm way_splits.zip
```

### Download Pretrained Models

## Usage

For testing use `test.sh` with the downloaded pretrained model. 
For training use `train.sh` and edit the parser arguments as necessary

