# Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training
[ICML2020] Xuxi Chen*, Wuyang Chen*, Tianlong Chen, Ye Yuan, Chen Gong, Kewei Chen, Zhangyang Wang

# Set-up
## Environment
```
conda install pytorch==0.4.1 cuda90 torchvision -c pytorch
conda install matplotlib scikit-learn tqdm
pip install opencv-python
```
## Preparing Data
Download cifar-10 and extract it into `cifar/`. 

# Training
## Baseline
### MNIST
```python
python train.py --self-paced False --mean-teacher False 
```

### CIFAR-10
```python
python train.py --self-paced False --mean-teacher False --dataset cifar --datapath cifar
```
## Self-PU (without self-calibration)
Training with self-calibation would be expensive. A cheap alternative:
### MNIST

```python
python train_2s2t.py 
```
### CIFAR-10
```python
python train_2s2t.py --dataset cifar --datapath cifar --lr 5e-4  
```

## Self-PU 
Training with self-calibation would be expensive. A cheap alternative:
### MNIST

```python
python train_2s2t.py 
```
### CIFAR-10
```python
python train_2s2t.py --dataset cifar --datapath cifar --lr 5e-4  
```

# Results
## Reproduce



