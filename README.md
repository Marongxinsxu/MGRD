## MGRD: Multi-granularity Reconstruction Deviation Modeling for Time Series Anomaly Detection

## Preparation

Our code is based on PyTorch 1.10.0 and runnable for both windows and server. Required python packages:

> + python==3.7.12
> + numpy==1.21.6
> + torch==1.10.0
> + pyYAML==6.0

## Usage

The datasets are all placed in the data directory. \
For the SWaT and WADI dataset, you can apply for it by following its official tutorial:https://itrust.sutd.edu.sg/itrust-labs_datasets/



run:

```python main.py -model_config MGRD_MSL.yaml```
```python main.py -model_config MGRD_SMD.yaml```
```python main.py -model_config MGRD_SMAP.yaml```
```python main.py -model_config MGRD_SWaT.yaml```
```python main.py -model_config MGRD_WADI.yaml```

## Cite

Please cite our paper if you use this code in your own work.




## Report a problem

