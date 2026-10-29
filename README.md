<img width="995" height="860" alt="image" src="https://github.com/user-attachments/assets/7c98b8b7-772c-41d6-8b0a-a53517caedfb" /># Data Extrapolation for Text-to-image Generation on Small Datasets

Official Pytorch implementation for our paper [Regularized Wasserstein Distance for Discrete Generative Modeling and Clustering Analysis](https://arxiv.org/abs/2410.01638) 


<img width="523" height="168" alt="image" src="https://github.com/user-attachments/assets/981fdcee-8166-49dc-a145-1aeb977820eb" />



---
### Requirements
- python 3.12.3
- Pytorch 2.2.2
- RTX 3090 or stronger GPUs(~7 days on CUB and Oxford with 2 3090 TI, ~30 days on COCO with 2 3090 TI)

### Installation

Clone this repo.
```
git clone [https://github.com/senmaoy/generative-discrete-clustering.git]
cd RAT-Diffusion
conda env create -f environment.yml
conda activate RAT
