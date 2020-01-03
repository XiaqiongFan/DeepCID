# DeepCID
Raman spectra contain abundant information from molecules but are difficult to analyze, especially for the mixtures. Deep-Learning-Based Components Identification for Raman Spectroscopy (DeepCID) has been proposed for the problem of components identification. Convolution Neural Network (CNN) models have been established to predict the presence of the components in the mixtures.

<div align="center">
<img src="https://raw.githubusercontent.com/xiaqiong/DeepCID/master/Flowchart_DeepCID.jpg" width=403 height=316 />
</div>

# Installation
## Install Python

Python 3.6 is recommended.

[python](https://www.python.org)

## Install tensorflow

[tensorflow](https://www.tensorflow.org)

## Install dependent packages

**1.Numpy**

pip install numpy

**2.Scipy**

pip install Scipy

**3.Matplotlib**

pip install Matplotlib

# Clone the repo and run it directly

[git clone at：https://github.com/xiaqiong/DeepCID.git](https://github.com/xiaqiong/DeepCID.git) 

# Download the model and run directly

Since the model exceeded the limit, we have uploaded all the models and the  information of mixtures to the Baidu SkyDrive and Google driver.

[Download at: Baidu SkyDrive](https://pan.baidu.com/s/1I0WMEvKvPNicy-i4Ru6uHQ) or [Google driver](https://drive.google.com/drive/folders/1DzMqiJRPDaLn2PcFW_myY_p0PO_VVEpS?usp=sharing)

**1.Training your model**

Run the file 'one-component-model.py'.The corresponding example data have been uploaded at [https://pan.baidu.com/s/1Q0QeHirzmBXVJZVqBx58Bw](https://pan.baidu.com/s/1Q0QeHirzmBXVJZVqBx58Bw) 

**2.Predict mixture spectra data**

Run the file 'DeepCID.py'.An example mixture data have been uploaded at Baidu SkyDrive (named  'mixture.npy', 'label.npy' and 'namedata.csv').Download the model and these example data，DeepCID can be reload and predict easily.

# Paper
[Paper](https://pubs.rsc.org/en/content/articlehtml/2019/an/c8an02212g)

# Contact

Zhi-Min Zhang: zmzhang@csu.edu.cn


