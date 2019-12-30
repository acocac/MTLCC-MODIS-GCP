
#!/bin/bash

YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
#YEARS=(2015)
#PROJECTS=(tile_1_463 tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713)
PROJECTS=(tile_0_563)
psize=(384)
RESOLUTIONS=(250m_spectral 250m_aux 500m_spectral) #250m_spectral 250m_aux 500m_spectral
#RESOLUTIONS=(250m_aux) #250m_spectral 250m_aux 500m_spectral
storage=(GCloud)
bucket=('gcptutorials')

for project in ${PROJECTS[@]}; do
    mkdir -p "F:/acoca/research/gee/dataset/$project/_logs"
    for year in ${YEARS[@]}; do
        for resolution in ${RESOLUTIONS[@]}; do
            logfname="F:/acoca/research/gee/dataset/${project}/_logs/$year.log"

            if [ "$resolution" = "500m_spectral" ]; then
                psize_500m=$(($psize / 2))

                echo "Downloading data for project $project and year $year and resolution $resolution with patchsize of $psize_500m"

                python 0_query_data.py \
                    -o=$project \
                    -p=$psize_500m \
                    -y=$year \
                    -r=$resolution \
                    -s=$storage \
                    -b=$bucket > $logfname 2>&1

            else
                psize_250m=$psize

                echo "Downloading data for project $project and year $year and resolution $resolution with patchsize of $psize"

                python 0_query_data.py \
                    -o=$project \
                    -p=$psize_250m \
                    -y=$year \
                    -r=$resolution \
                    -s=$storage \
                    -b=$bucket > $logfname 2>&1
            fi
        done
    done
done