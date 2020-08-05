## SpTSkM
An demo of SpTSkM for long-term re-ID. The code is tested on Anaconda python 3.6, Pytorch 0.4.
## Data Preparation
* Download MARS [1] from http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html.
* Download the data partition information from https://github.com/liangzheng06/MARS-evaluation.
* We use pretrained LIP [2] to extract the image mask, LIP is available on https://github.com/Engineering-Course/LIP_JPPNet.
* For 3D skeleton estimation, we use a weakly-supervised method [3] from https://github.com/xingyizhou/pose-hg-3d. 
## Requirement
* Python 3.6
* Pytorch
* OpenCV
* CUDA and cudnn
## Method
The framework of SpTSkM as shown in the following figure. It includes two streams, SSIN and SMIN. SSIN takes image squences and their corresponding masks as input.It aims to learn motion patterns, shape information and some subtle identity properties. SMIN is adapted from ST-GCN [4], which aims to learn pure motion pattern from skeleton squences.

![framework](/imgs/framework.png)

## Training
* For SSIN, place data to ```/data/mars``` and run ```CUDA_VISIBLE_DEVICES='0' python train_adam.py``` in ```/SSIN/```.
* For SMIN, prepare data by running ```python prepare_mars.py``` in the folder ```/SMIN/data/```, and running ```   CUDA_VISIBLE_DEVICES='0' python main.py recognition -c /config/st_gcn/Mars/train.yaml``` in ```/SMIN/```.
## Evaluation
During testing phase, extracting features as following,
* For SSIN, run ```python train_adam.py --evaluate --ckpt_path='./logs/best_model.py'```
* For SMIN, run ```python main.py recognition -c /config/st_gcn/Mars/test.yaml``` in ```/SMIN/```.

## References
[1] Zheng, L., Bie, Z., Sun, Y., Wang, J., Su, C., Wang, S., & Tian, Q. (2016, October). Mars: A video benchmark for large-scale person re-identification. In European Conference on Computer Vision (pp. 868-884).
[2] Liang, X., Gong, K., Shen, X., & Lin, L. (2018). Look into person: Joint body parsing & pose estimation network and a new benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(4), 871-885.
[3] Zhou, X., Huang, Q., Sun, X., Xue, X., & Wei, Y. (2017). Towards 3d human pose estimation in the wild: a weakly-supervised approach. In Proceedings of the IEEE International Conference on Computer Vision (pp. 398-407).
[4] Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. In Thirty-second AAAI conference on artificial intelligence, 7442-7452.



