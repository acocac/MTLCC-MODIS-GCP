
#!/bin/bash

YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
PROJECTS=(tile_0_563)
psize=(384)

for project in ${PROJECTS[@]}; do
    mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs"
    for year in ${YEARS[@]}; do

        if [ "$year" = 2001 ]; then
            timesteps=(45)
        elif [ "$year" != 2001  ]; then
            timesteps=(46)
        fi

        echo "Processing data for project $project and $year"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/$year.log"
        python 0_merge_multisources.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --psize=$psize \
            --tyear=$year \
            --timesteps=$timesteps > $logfname 2>&1

    done
done