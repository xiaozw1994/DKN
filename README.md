# DKN MTSC
## Setup
Configure the experimental environment and tools listed below: 
```
pip3 install -r require.txt
```
### Dataset
Download UEA datasets
```
URL: http://timeseriesclassification.com/dataset.php
```
When obtained the UEA zip file, one should unzipthis file and add to:
```
repo:  ./data/
``` 
## Implementation
### Global Configuration 
```
see: ./processing/data.py
####configure introduce:
####dataset repo
save_file =  "/mnt/54b93d63-868e-4b2c-aa09-c9109c3e67be/multivariateTSC/UCE"
#### GPU index used
GPU_number = 0
#### dataset name
dataset_name = "Beef"
###  coeffecient for loss function
loss_coefficient = 0.1
### a scalarable temporate for distillation
temperate = 1.0
#####  the weight of dual non-target class loss
beta = 1.0
##### the weight of dual target class loss
alpha = 1.0
###### learning rate
learning_rate = 0.0001
##### weight_decay value
weight_decay = 0.0005
#### training epoch
epoch = 200
#### save path
save_path =  "save_model/"
```
### Run Example
```
python3 main.py -n 1 -b 1 -r 0.01 -s 50 -p 1 -e 500
```
