# About the APP

### How it works

#### Local group
Select a region with zoom N and convolves it select points of interest. Each of the square box in the image bellow are subregions selected for classification.

![img_2.png](https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/img_2.png)

Each one of those square are subdivide in N tiles, these tiles them feed a CNN model, and based on its position new areas are selected. This step is repeated untill no more areas nearby are found.

![img_1.png](https://github.com/PedroFrias/amazonian_rainforest_survey/blob/main/imgs/img_1.png)

Note that it may result in redundancy, which is avoided with a nearest neighboor algorithm.
