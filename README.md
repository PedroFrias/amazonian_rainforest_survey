###### Project under development (June 15, 2022)

# About the APP
### How it works
<p align="justify">
Based on satelite imagens colected from googlemaps API, regions are selected for its content classification. The main execution is divided in four-ish steps: (step 1) requests a image (I) centered ant lat/lon with googlemaps API, then the image is convolved (step 2) to extract features, mainly high contrast with the surroundings, this features creates regions (R) that are deamed worth to be classified later. Each R is composed by surveral points, and by clusttering (step 3) them its possible to get thier geometric center (X, Y), which will used for tilling I to feed a Convolutional Neural Network (CNN) (step 4). Finally (step 5), the propagation is calculated, if anything was found previously.

With this routine is possible to cover a large area with minimal data and labour - steps number 2, 3 and 5 are used as performance enhancers, being steps 2 an 3 responsible to minimazing the amount of iterations necessary to achieve the goal, and step number 5 retro feeds 1, only and only if, the new __I__ have analytical sense to it.
</p>

<p align="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/images/cnn_deforestation_diagram.png">
</p>


# How to run

<p align="justify">
First you'll need to get a [API key](https://console.cloud.google.com/apis). This service is paid, but you can get around 27000 imagens with the 200 credits provided by Google monthly. Obs.: these credits don't stack!
</p>

(The following instructions apply to Posix/bash. Windows users should check [here](https://docs.python.org/3/library/venv.html).)
<p align="justify">
Clone this repository and open a terminal inside the root folder, create and activate a new virtual environment (recommended) by running the following:
</p>

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

<p align="justify">
Marked as red are the sites of deforestation found at coordinates lat -3.227735 and lon -60.849749 (June, 15 2020). 
The propagation is show in the footer.
</p>
<p align="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/Figure_1.png">
</p>
