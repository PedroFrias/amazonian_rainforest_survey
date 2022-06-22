###### Project under development (June 15, 2022)

# About the APP
### How it works
#### 1. Classification
<p align="justify">
 Initally a satellite image (I) is requested from Google Maps' API. I have a twofold meaning: 1) feed a Convolutional Neural Network; 2) serves as a iterator, by convolving it to extract feature, the locations of this features are used to select tiles - with this aproach only the locations with meanginfull values are processed.
  
The workflow bellow shows the process:
</p>

<p align="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/images/cnn_classification_diagram.png">
</p>


# How to run

<p align="justify">
First you'll need to get a API key, get one on https://console.cloud.google.com/apis. This service is paid, but you can get around 27000 imagens with the 200 credits provided by Google monthly. Obs.: these credits don't stack!
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
# Results (so far...)

<p align="justify">
Graph 1.: CNN main execuution on lat -2.934369 and lon -59.478271 (June, 18 2020).
 
Graph 2.: Results(blue) and predictions(orange).
</p>
<p float="center">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/images/cnn_classifying_data.gif" height="265">
  <img src="https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/images/results_and_predictions.png" height="265">
</p>
