###### Project under development (June 10, 2022)

# About the APP
### How it works
Based on satelite imagens colected from googlemaps API, regions are selected for content classification. The main execution is divided in two parts: local and large scale survey.

#### Large scale survey
Generate a heatmap where each square is as pseudo-probability - achieved by convolving and pooling(max) - of a inner region having been deforested. This is done mainly to reduce the number of operations preformed by methods futher down the line, since each region works as a separeted entity.

![Figure_1.png](https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/Figure_1.png)

A sample of the regions with with potential is selected for classificatio, and based on it's results the tiles adjacent can be either be activated or diactivated.

#### Small scale survey
Each one of those square are subdivide in N tiles, these tiles them feed though CNN model, and based on its position new areas are selected. This step is repeated untill no more areas nearby are found. Note that it may result in redundancy, which is avoided with a nearest neighboor algorithm.

<p align="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/Figure_2.png">
</p

## How to run this app
First you'll need to get a [API key](https://console.cloud.google.com/apis). This service is paid, but you can get around 27000 imagens with the 200 credits provided by Google monthly. Obs.: these credits don't stack!

(The following instructions apply to Posix/bash. Windows users should check [here](https://docs.python.org/3/library/venv.html).)
Clone this repository and open a terminal inside the root folder, create and activate a new virtual environment (recommended) by running the following:
```
python3 -m venv myvenv
source myvenv/bin/activate
```
Install the requirements:
```
pip install -r requirements.txt
```
Run the app:
```
python main.py
```
