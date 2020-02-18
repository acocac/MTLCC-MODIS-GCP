
#!/bin/bash

YEARS=(2001 2002 2003)
PROJECTS=(tl_bogota)
psize=(384)
RESOLUTIONS=(250m_spectral 250m_aux 500m_spectral) #250m_spectral 250m_aux 500m_spectral
storage=(Gdrive)
bucket=('thesis-2020')

for project in ${PROJECTS[@]}; do
    mkdir -p "F:/acoca/research/gee/dataset/$project/_logs"
    for year in ${YEARS[@]}; do
        for resolution in ${RESOLUTIONS[@]}; do
            logfname="F:/acoca/research/gee/dataset/${project}/_logs/${year}_download.log"

            if [ "$resolution" = "500m_spectral" ]; then
                psize_500m=$(($psize / 2))

                echo "Downloading data for project $project and year $year and resolution $resolution with patchsize of $psize_500m"

                python 0_query_data_dl.py \
                    -o=$project \
                    -p=$psize_500m \
                    -y=$year \
                    -r=$resolution \
                    -s=$storage \
                    -b=$bucket > $logfname 2>&1

            else
                psize_250m=$psize

                echo "Downloading data for project $project and year $year and resolution $resolution with patchsize of $psize"

                python 0_query_data_dl.py \
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