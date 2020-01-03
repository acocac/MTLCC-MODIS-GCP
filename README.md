## MTLCC-MODIS-GCP
# Multitemporal Land Cover Classification Network (MTLCC) adapted to read MODIS data using tf.Estimator and Google AI SDK

The MTLCC network is a state-of-the-art recurrent neural network approach to encode multi-temporal data for land cover classification.

##### Source code of Rußwurm & Körner (2018)

If you use this model consider citing
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. ISPRS International Journal of Geo-Information 7(4),29. ![DOI](https://doi.org/10.3390/ijgi7040129)

```

## Contributions of this repository
* The original `Tensorflow 1.7` repository located at `https://github.com/TUM-LMF/MTLCC` was forked and adapted to TF 1.14 using tf.Estimator and Google AI structure.
* The MTLCC network is adapted to read MODIS 250m/500m surface reflectance data.

## Instructions
* Scripts for training and evaluation are provided.
* The code can be executed after downloading the demo data (available [here](https://drive.google.com/drive/folders/1ljxThnqgeNsnfv_qejI-jE8O4bH9mHkW?usp=sharing)
*  After installing the dependencies the python scripts should be executable.

## Dependencies
Implementations of ConvGRU and ConvLSTM was adapted from https://github.com/carlthome/tensorflow-convlstm-cell and adapted into the trainer/utils.py script.

## Acknowledgements
The author acknowledges Marc Rußwurm for his constant feedback of this implementation derived from the original repository.

## Contributing
Contributions via pull requests are welcome. Please make sure that changes pass the unit tests. Any bugs and problems can be reported on the repo's issues page.

Python packages
```bash
conda install -y gdal
pip install tensorflow-gpu=1.14
pip install pandas
pip install configparser
pip install --upgrade google-api-python-client
```
## Network training and evaluation

### on local machine (requires dependencies installed)

#### train the network graph for 24px tiles
```bash
bash bin/run.train.local.sh
```

## Monitor training/validation curves
### on local machine (requires dependencies installed)

```bash
tensorboard --logdir=.
```
