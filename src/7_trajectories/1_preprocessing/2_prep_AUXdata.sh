#!/bin/bash

project=(AMZ)
years="2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018"
nworkers=(0)

logdir="F:/acoca/research/gee/dataset/${project}/trajectories/_logs/"
mkdir -p $logdir

echo "Creating AUX data for year(s): $years"
logfname="${logdir}/auxdata.log"

python 2_prepare_AUX_byyear.py \
        --preddir="F:/acoca/research/gee/dataset/${project}/implementation/ancillary" \
        --auxdir="F:/acoca/research/gee/dataset/${project}/implementation" \
        --outdir="F:/acoca/research/gee/dataset/${project}/trajectories/ep30/aux" \
        --targetyear $years \
        --nworkers $nworkers \
    	--noconfirm > $logfname 2>&1