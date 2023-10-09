# XYW-Net
Bio-inspired XYW parallel pathway edge detection network
# Please Waiting.
****
All results is evaluated on Python 3.8 with PyTorch 1.11.0+cuda113 and MATLAB R2018b.
We only publish our test results on the BSDS500, NYUDv2,BIPEDv1 and Multicue datasets for now.The implementation details of code will be updated after the paper is officially published.
****
# Datasets
We use the links in RCF Repository (really thanks for that).
The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:

    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
Multicue Dataset is Here

    https://drive.google.com/file/d/1-tyt_KyzlYc9APafdh5mHJzh2K_F2hM8/view?usp=sharing
BIPED Dataset is Here

    https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM
****
# Tools
The evaluation program of ODS OIS is here:

    https://github.com/pdollar/edges
The PR curve tool is here:

    https://github.com/MCG-NKU/plot-edge-pr-curves
****
# Reference
When building our code, we referenced the repositories as follow:

[LMGCN](https://github.com/cimerainbow/LMGCN)

[PidiNet](https://github.com/zhuoinoulu/pidinet)
