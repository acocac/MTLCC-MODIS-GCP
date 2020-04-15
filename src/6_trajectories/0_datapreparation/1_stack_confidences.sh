#!/bin/bash

project=(AMZ)
YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)

logdir="F:/acoca/research/gee/dataset/${project}/prediction/_logs/"
mkdir -p $logdir

for year in ${YEARS[@]}; do

    echo "Creating LC confidences stack for year: $year"
    logfname="${logdir}/${year}_cfstack.log"

    python 1_stack_confidences.py \
            --indir="F:/acoca/research/gee/dataset/${project}/prediction/confidences/${year}" \
            --outdir="F:/acoca/research/gee/dataset/${project}/prediction/confidences_stack" \
            --targetyear $year > $logfname 2>&1
done