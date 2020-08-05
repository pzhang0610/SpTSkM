# SpTSkM
An demo of SpTSkM for long-term re-ID. The code is tested on Anaconda python 3.6, Pytorch 0.4.
## Data Preparation
* Download MARS [1] from http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html.
* Download the data partition information from https://github.com/liangzheng06/MARS-evaluation.
* We use pretrained LIP [2] to extract the image mask, LIP is available on https://github.com/Engineering-Course/LIP_JPPNet.
* For 3D skeleton estimation, we use a weakly-supervised method [3] from https://github.com/xingyizhou/pose-hg-3d. 

## Method
The framework of SpTSkM as shown in the following figure. It includes two streams, SSIN and SMIN. SSIN takes image squences and their corresponding masks as input.It aims to learn motion patterns, shape information and some subtle identity properties. SMIN is adapted from ST-GCN [4], which aims to learn pure motion pattern from skeleton squences.
![Framework](/imgs/framework.eps)

## Training

## Testing
