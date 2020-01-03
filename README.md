## MTLCC-MODIS-GCP
### Multitemporal Land Cover Classification Network (MTLCC) adapted to MODIS data using tf.Estimator and Google AI SDK

The MTLCC network is a state-of-the-art recurrent neural network approach to encode multi-temporal data for land cover classification.

##### Source code of Rußwurm & Körner (2018)

If you use this model consider citing:
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. ISPRS International Journal of Geo-Information 7(4),29. https://doi.org/10.3390/ijgi7040129

```

### Contributions of this repository
* The MTLCC network reads MODIS 250m/500m surface reflectance data and auxiliary data (labelled datasets, elevation, climate).
* These datasets can be downloaded from Google Earth Engine using GEE python API (see [0_downloaddata](https://github.com/acocac/MTLCC-MODIS-GCP/tree/master/0_downloaddata)) and preprocessed according to the scripts in [1_datapreparation](https://github.com/acocac/MTLCC-MODIS-GCP/tree/master/1_datapreparation). 
* The partitions for training the MTLCC model can be done using the scripts in [2_datapartition](https://github.com/acocac/MTLCC-MODIS-GCP/tree/master/2_datapartition).
* The original `Tensorflow 1.7` repository located at `https://github.com/TUM-LMF/MTLCC` was forked and adapted to TF 1.14 using tf.Estimator and Google AI structure. The structure of the scripts are according to Google AI as can be found in [3_model_mtlcc](https://github.com/acocac/MTLCC-MODIS-GCP/tree/master/3_model_mtlcc).
* Scripts for analysing the results and outputs of the training from Google AI SDK are available in [4_results](https://github.com/acocac/MTLCC-MODIS-GCP/tree/master/4_results).

### Instructions
* Scripts for training and evaluation are provided.
* The code can be executed after downloading the demo data (available [here](https://drive.google.com/drive/folders/1ljxThnqgeNsnfv_qejI-jE8O4bH9mHkW?usp=sharing)).
*  After installing the dependencies the python scripts should be executable.

### Acknowledgements
The author acknowledges Marc Rußwurm for a constant feedback of this implementation derived from his original repository. 

This development is part of my on-going PhD titled ["Modelling pan-tropical land cover and land-use change trajectories of newly deforested areas"](https://www.kcl.ac.uk/people/alejandro-coca-castro) at King's College London (KCL) supervised by Dr. Mark Mulligan (first supervisor at KCL) and Dr. Louis Reymondin (co-supervisor at the International Center for Tropical Agriculture - CIAT).

### Future work
* Enable TPU-processing, using TPUestimator.
* Improve strategies for multiGPU processing (currently only working with tf.distribute.MirroredStrategy).
* Fix serving functions for online predictions and integration with Google Earth Engine's ee.Model (further info [here](https://developers.google.com/earth-engine/tensorflow)).

### Contributing
Contributions via pull requests are welcome. Please make sure that changes pass the unit tests. Any bugs and problems can be reported on the repo's issues page.

### Network training and evaluation

#### on local machine (requires dependencies installed)

#### train the network graph for 24px tiles
```bash
bash bin/run.train.local.sh
```

### Monitor training/validation curves
#### on local machine (requires dependencies installed)

```bash
tensorboard --logdir=.
```

Python packages
```bash
conda install -y gdal
pip install tensorflow-gpu=1.14
pip install pandas
pip install configparser
pip install --upgrade google-api-python-client
```

### Dependencies
Implementations of ConvGRU and ConvLSTM was adapted from https://github.com/carlthome/tensorflow-convlstm-cell and adapted into the trainer/utils.py script.
