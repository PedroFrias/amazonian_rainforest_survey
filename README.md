###### Project under development (June 13, 2022)

# About the APP
### How it works
Based on satelite imagens colected from googlemaps API, regions are selected for its content classification. The main execution is divided in four-ish step:

1. Collecting data;
2. Convolution;
3. Clusttering;
4. Classification;
5. Propagation.

Where steps number 2, 3 and 5 are used as performance enhancers, being steps 2 an 3 responsible to minimazing the amount of iterations necessary to achieve the goal, and step number 5 retro feeds 1, only and only if, the new __I__ have analytical sense to it.

### 1. Collecting data:
requests a image (I) centered ant lat/lon with googlemaps API

### 2. Convolution:
This step couple with __step__ 3 reduce the amount of iterations by extracting features, mainly high contrast with the surroundings, this features creates regions (R) that are deamed worth to be classified.

### 3. Clusttering:
Each __R__ is composed for surveral points, by clusttering them its possible to get thier geometric center (X, Y).

### 4. Classification:
Select tiles from __I__ centered at __X__, __Y__ to feed through a Convolutional Neural Network (CNN).

### 5. Propagation:
If anything is found at 4 __R__ is reduced to a 3x3 version of itsef using MaxPooling (P), as __P__ is set to be the regions adjascent __I__.

With this routine is possible to cover a large area with minimal data and labour.


# How to run
 
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
# Results
Marked as red are the sites of deforestation found at coordinates lat -3.227735 and lon -60.849749 (June, 15 2020). 
The propagation is show in the footer.

<p align="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/Figure_1.png">
</p>
