rm(list = ls())

###source
#https://github.com/amysheep/NUTrajectoryPredictionAnalysis/blob/af2170541a8959cac6aff290b60023017f93496a/TrajectoryPredictionAnalysis.ipynb

###run outliers
findOutliers <- function(col,coef){ 
  cuartil.primero = quantile(col,0.05)
  cuartil.tercero = quantile(col,0.95)
  iqr <- cuartil.tercero - cuartil.primero
  
  extremo.superior.outlier <- cuartil.tercero + coef * iqr
  extremo.inferior.outlier <- cuartil.primero - coef * iqr
  
  return( which((col > extremo.superior.outlier) | (col < extremo.inferior.outlier)))
}

vector_claves_outliers_IQR_en_alguna_columna <- function(datos, coef=1.5){
  vector.es.outlier <- sapply(datos[1:ncol(datos)], findOutliers,coef)
  vector.es.outlier
}

computeOutliers <- function(data, type='remove', k=2, coef = 1.5){
  outliers <- vector_claves_outliers_IQR_en_alguna_columna(data, coef)
  if (type == 'remove'){
    index.to.keep <- setdiff(c(1:nrow(data)),unlist(outliers))
    return (index.to.keep)
  }
  else if(type == 'knn'){
    data.with.na <- changeOutliersValue(outliers,data, type='knn')
    return(computeMissingValues(data.with.na,type='knn',k=k))
  }
  else if(type == 'median'){
    return(changeOutliersValue(outliers,data))
  }
  else if(type == 'mean'){
    return(changeOutliersValue(outliers,data, type = 'mean'))
  }
  else if(type == 'rf'){
    data.with.na <- changeOutliersValue(outliers,data, type='rf')
    return(computeMissingValues(data.with.na,type='rf'))
  }
  else if(type == 'mice'){
    data.with.na <- changeOutliersValue(outliers,data, type='mice')
    return(computeMissingValues(data,type='mice'))
  }
  
  return(data) # es necesario?
}

changeOutliersValue <- function(outliers,data,type = 'median'){
  i = 1
  j = 1
  
  n = ncol(data)
  
  while(j <= n){
    outliers_columna = outliers[[j]]
    m = length(outliers_columna)
    
    while(i <= m){
      if (type == 'median'){
        data[outliers_columna[i],j] = median(data[,j], na.rm = TRUE)
      }
      else if(type == 'mean'){
        data[outliers_columna[i],j] = mean(data[,j], na.rm = TRUE)
      }
      else {
        data[outliers_columna[i],j] = NA
      }
      
      i = i +1
    }
    
    i = 1
    j = j + 1
  }
  return(data)
}

library(compareGroups)
library(caret)
library(raster)
library(doParallel)
library(dplyr)
library(snowfall)
library(labelled)
library(viridis)
library(Hmisc)
library(hrbrthemes)
library(ggpubr)

tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatoryv2")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clustersv2")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distancesv2")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_datav2")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)
charts_dir <- paste0("E:/acocac/research/",tile,"/trajectories/chartsv2")
dir.create(charts_dir, showWarnings = FALSE, recursive = T)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
proj <- CRS('+proj=longlat +ellps=WGS84')

###start process
fn <- 'WARD_clustersk6_2004-2016_minperiod12_TRATE_OM_target.RSav'

file_name <- paste0('auxdata_',fn)
file_path <- paste0(explanatory_dir,'/',file_name)

if(file.exists(paste0(file_path))){
  load(file_path)
} else{
  load(paste0(clusters_dir,'/',fn))

  access <- raster(paste0(aux_dir,'/ancillary/processed/external/access.tif'))
  dem <- raster(paste0(aux_dir,'/ancillary/processed/external/srtm.tif'))
  slope <- raster(paste0(aux_dir,'/ancillary/processed/external/slope.tif'))
  precipitation <- raster(paste0(aux_dir,'/ancillary/processed/external/bio12.tif'))
  pas_conservation <- raster(paste0(aux_dir,'/ancillary/processed/external/distance_PAs_conservation_AMZ.tif'))
  pas_exploitation <- raster(paste0(aux_dir,'/ancillary/processed/external/distance_PAs_exploitation_AMZ.tif'))

  c1 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c1.tif'))
  c2 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c2.tif'))
  c3 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c3.tif'))
  c4 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c4.tif'))
  c5 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c5.tif'))
  c6 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c6.tif'))

  ##extract raster values
  ##add and prepare aux columns
  tab.target_geo = unique_seq_subset
  coordinates(tab.target_geo) <- c("x", "y")
  mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))

  #external
  dem_val =raster::extract(dem, mypoints)
  slope_val =raster::extract(slope, mypoints)
  access_val =raster::extract(access, mypoints)
  prec_val =raster::extract(precipitation, mypoints)
  pascon_val =raster::extract(pas_conservation, mypoints)
  pasexp_val =raster::extract(pas_exploitation, mypoints)

  #internal
  c1_val =raster::extract(c1, mypoints)
  c2_val =raster::extract(c2, mypoints)
  c3_val =raster::extract(c3, mypoints)
  c4_val =raster::extract(c4, mypoints)
  c5_val =raster::extract(c5, mypoints)
  c6_val =raster::extract(c6, mypoints)

  #external values
  tab.target_geo$access = (access_val)/(60*24)
  tab.target_geo$dem = dem_val
  tab.target_geo$slope = slope_val
  tab.target_geo$prec = prec_val
  tab.target_geo$pascon = (pascon_val * 231.91560544825498) / 1000
  tab.target_geo$pasexp = (pasexp_val * 231.91560544825498) / 1000

  #internal values
  tab.target_geo$c1 = (c1_val * 231.91560544825498) / 1000
  tab.target_geo$c2 = (c2_val * 231.91560544825498) / 1000
  tab.target_geo$c3 = (c3_val * 231.91560544825498) / 1000
  tab.target_geo$c4 = (c4_val * 231.91560544825498) / 1000
  tab.target_geo$c5 = (c5_val * 231.91560544825498) / 1000
  tab.target_geo$c6 = (c6_val * 231.91560544825498) / 1000


  save(tab.target_geo, file=paste0(file_path))
}

tab.target_df = as.data.frame(tab.target_geo)

group <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("t1", "t2", "t3", "t4", "t5", "t6")
)

final_df = data.frame(group, tab.target_df[,17:28])

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

final_df = data.frame(group, tab.target_df[,c(explanatory_int,explanatory_ext)])

gooddata = computeOutliers(tab.target_df[,c(explanatory_ext,explanatory_int)], type = 'remove')
good_df_q95 = final_df[gooddata,]

df_wnoise = good_df_q95

##charts###
preProcValues <- preProcess(df_wnoise[-1], method = c("range"))
#preProcValues <- preProcess(final_df[-1], method = c("center", "scale", "BoxCox"))

trainTransformed <- predict(preProcValues, df_wnoise[-1])

boxplot_df <- cbind(group=df_wnoise$group,trainTransformed)

target_reshape <- melt(boxplot_df, id = c("group"), value.name = "value")

p1<- ggplot(target_reshape[target_reshape$variable %in% (explanatory_int),], aes(variable, value, fill=factor(group))) +
  geom_boxplot(outlier.shape = NA) + 
  theme_ipsum_rc(axis_title_size = 16, base_size=13) +
  labs(x="Internal variables", y="Normalised Values [0-1]") + 
  scale_fill_manual(values = c('#377eb8', '#a65628', '#4daf4a', 
                               '#ff7f00','#f781bf', '#984ea3'),
                    labels = postlossLC) +
  scale_x_discrete(labels = etiquettes_p1) +
  theme(legend.text=element_text(size=15), legend.title = element_blank(), legend.spacing.x = unit(1.0, 'cm')) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank()) +
  guides(fill = guide_legend(nrow = 2))

p2<- ggplot(target_reshape[target_reshape$variable %in% (explanatory_ext),], aes(variable, value, fill=factor(group))) +
  geom_boxplot(outlier.shape = NA) + 
  theme_ipsum_rc(axis_title_size = 16, base_size=13) +
  labs(x="External variables", y="Normalised Values [0-1]") + 
  scale_fill_manual(values = c('#377eb8', '#a65628', '#4daf4a', 
                               '#ff7f00','#f781bf', '#984ea3'),
                    labels = postlossLC) +
  scale_x_discrete(labels = c("Accessibility","Elevation","Slope","Precipitation","Distance to \nconservation PAs", "Distance to \nexploitation PAs")) +
  theme(legend.text=element_text(size=15), legend.title = element_blank(), legend.spacing.x = unit(1.0, 'cm')) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank()) +
  guides(fill = guide_legend(nrow = 2))

final_plot <- ggarrange(p1, p2, legend = "top", common.legend = TRUE, ncol=1, nrow=2)
final_plot <- annotate_figure(final_plot, top = text_grob("Post-loss LC trajectory", size = 17, face = "bold"))

png(file = paste0(charts_dir,"/extint_postloss_wooutlier_q95_pointsNA.png"), width = 1700, height = 1200, res = 150)
print(final_plot)
dev.off()

##random forest and cubist
#====================================================================================== -

library(caret)
library(ranger)
library(tidyverse)

path_to_utils <- "E:/acocac/research/scripts/MTLCC-MODIS-GCP/src/7_trajectories/2_sequences/v2/" #Specify path to functions

source(paste0(path_to_utils, "005_rfe_utils.R"))

#====================================================================================== -

data <- df_wnoise

###additional tests##
#excluding T1
data = data[data$group != 't1',]
data$group <- factor(data$group)

#separate variables
explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

data = data.frame(data$group, data[,explanatory_ext])
#

# The candidate set of the number of predictors to evaluate
subsets <- c(length(data),10,8,6,4,2)

classifiers <- c('ranger')
CV = 15

for (c in classifiers){
  print(c)
  # Set up a cluster
  no_cores <- detectCores() - 1
  
  cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
  registerDoParallel(cl)                #register the cluster
  
  set.seed(40)
  
  # Perform recursive feature elimination
  rfe <- perform_rfe(response = "group", base_learner = c, type = "classification",
                     p = 0.8, times = CV, 
                     subsets = subsets, data = data,
                     importance = "permutation",
                     num.trees = 100)
  
  file_name <- paste0("rfe_",c,"_",CV,"CV_target_all_excT1.RData")
  file_path <- paste0(explanatory_dir,'/',file_name)
  
  save(rfe, file=paste0(file_path))
  
  stopCluster(cl = cl)
}

#====================================================================================== -
c <- 'ranger'
CV <- 30

file_name <- paste0("rfe_",c,"_",CV,"CV_target_all.RData")
file_path <- paste0(explanatory_dir,'/',file_name)
load(paste0(explanatory_dir,'/',file_name))

out <- tidy_rfe_output(rfe, "RF")
PROF <- plot_perf_profile(out[[1]])

p1_multinom <- PROF
p2_rf <- PROF

final_plot <- ggarrange(p1_multinom, p2_rf, legend = "top", common.legend = TRUE, ncol=2, nrow=1)
#final_plot <- annotate_figure(final_plot, top = text_grob("Post-loss LC trajectory", size = 17, face = "bold"))

png(file = paste0(charts_dir,"/RFE_comparison_all.png"), width = 1700, height = 1200, res = 150)
print(final_plot)
dev.off()



