# SpTSkM
An demo of SpTSkM for long-term re-ID. The code is tested on Anaconda python 3.6, Pytorch 0.4.
## Data Preparation
* Download MARS [1] from \url{http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html}.
* Download the data partition information from \url{https://github.com/liangzheng06/MARS-evaluation}.
* We use pretrained LIP [2] to extract the image mask, LIP is available on \url{https://github.com/Engineering-Course/LIP_JPPNet}.
* For 3D skeleton estimation, we use a weakly-supervised method [3] from \url{https://github.com/xingyizhou/pose-hg-3d}.
