## Prediction of random fields of mechanical properties from microstructure images

implementation based on https://www.sciencedirect.com/science/article/pii/S0045782524004638 + additional features

### Input data:
#### training data
- Dataset of 2D microstructure images (InputData/10000_Images_L_50_Lw_50_radius_2.5_res_128.mat)
  - referred to as stochastic volume elements (SVEs)
  - resolution 128 $\times$ 128 pixels
  - dimensions $L \times L = 50\times 50$, fiber radius $r = 5$, varying volume fraction from 0 to 50%
- corresponding effective properties $C_{ij}$ (InputData/10000_Params_Stiffness_ratio_100_Lw_50.txt)
- corresponding volume fraction $v_f$ (InputData/10000_VF_Stiffness_ratio_100_Lw_50.txt) for visualization

#### testing data
- Dataset of large microstructure images, from which random fields are extracted (InputData/200_Images_L500_LW50_radius_2.5_res_1280.mat)
  - resolution 1280 $\times$ 1280 pixels
  - dimensions  $L \times L = 500\times 500$, fiber radius $r = 5$, varying volume fraction from 0 to 50%
  - images were processed with a moving window technique using a window size $L_w = 50$ at a step $DL_w = L_w/4 =12.5$
- corresponding random fields of effective properties (OutputData/TrueCij)
  - computed at each moving window position (1444 points per image)
 
### Features
The present code contains 4 different applications, based on the above data. The first two are included in the aforementioned journal paper.

**1)**  Prediction of effective properties of microstructure image using a CNN (CNN_train.py)
<p align="center">
<img width="400"  alt="cnn" src="https://github.com/user-attachments/assets/8e268f3d-2b46-42bb-b677-675d1c9f8f6f" />
</p>





**2)**  Prediction of random fields of mechanical properties of a microstructure image (CNN_predict.py)
   - requires training via (CNN_train.py)
<p align="center">
<img width="500"  alt="RF" src="https://github.com/user-attachments/assets/ddd63bf1-0f1a-472b-9b5f-0008c1c43aa3" />
</p>






**3)**  Generation of microstructure images based on existing image dataset using a variational autoencoder (VAE.py)

**4)**  Prediction of effective properties of microstructure image using a GNN (GNN.ipynb) 
   - requires additional file of fiber locations for each image (InputData/10000_Fiber_centers_L_50_radius2.5_res_128.mat)
  

#### Note:
- 1),2) and 3) were implemented using Keras (tensorflow), 4) is implemented using pytorch

- Training was initially performed on local gpu and GPU clusters. Google Colab is used for later implementations, hence  some data is loaded via google drive
