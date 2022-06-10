# About the APP
#
### How it works
Based on satelite imagens colected from googlemaps API, regions are selected for content classification. The main execution is divided in two parts: local and large survey.

#### Local survey
Select a region with zoom N and convolves it select points of interest. Each of the square box in the image bellow are subregions selected for classification.

![img_2.png](https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/img_2.png)

Each one of those square are subdivide in N tiles, these tiles them feed a CNN model, and based on its position new areas are selected. This step is repeated untill no more areas nearby are found. Note that it may result in redundancy, which is avoided with a nearest neighboor algorithm.

![img_1.png](https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/img_1.png)

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
