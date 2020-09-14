# Drug Solubility Prediction based on calculated molecular descriptors via Mordred

This is a collection neural network model to predict aqueous drug-like molecules based molecular descriptors calculated by Mordred (https://github.com/mordred-descriptor/mordred). Mordred has the capability to 1613 kinds of molecular descriptors. Several models are built based on the selection of molecular descriptors. 5 sets of molecular descriptors are selected for Pearson Correlation Coefficient between molecular descriptors and aqueous solubility larger than 0.8, 0.7, 0.5, 0.3 and 0.0 separately.


![image](https://github.com/jeah-z/Drug-Solubility-Prediction-Mordred/blob/master/Figures/dnn.png)
Fig. 1 Architecture of solubility prediction models based on DNN
![image](https://github.com/jeah-z/Drug-Solubility-Prediction-Mordred/blob/master/Figures/gcn.png)
Fig. 2 Architecture of solubility prediction models based on DNN

# How to use

First the descriptors can be calculated with the command line below:

```
python descriptor_cal.py
```

Then the dataset needs to be separated into training set and validation set with the command below:
```
python preprocess.py
```

The DNN models and GCN models can be trained with below commands separately:

```
DNN: python DNN.py
GCN: python train.py --model sch --epochs 20000 --dataset huus
```

