# GCN-HGT
Prediction of horizontal gene transfer (HGT) using graph convolutional network

This pipeline performs data splitting (train/validation/test) of computed HGT data, and trains a graph convolutional network model to predict HGT events.

* Hao Zhou, Juan F. Beltrán, Ilana L. Brito. [Functions predict horizontal gene transfer and the emergence of antibiotic resistance. Science Advances](https://www.science.org/doi/10.1126/sciadv.abj5056) (2021).

# Acknowledgement
* Kipf, Thomas N., and Max Welling. "Variational graph auto-encoders." arXiv preprint arXiv:1611.07308 (2016).
* Leskovec, Jure, and Rok Sosič. "Snap: A general-purpose network analysis and graph-mining library." ACM Transactions on Intelligent Systems and Technology (TIST) 8.1 (2016): 1-20.

# Requirement
* TensorFlow (1.13.1)
* python 3.5.5
* networkx
* scikit-learn
* scipy
* pandas
* numpy
* tqdm
