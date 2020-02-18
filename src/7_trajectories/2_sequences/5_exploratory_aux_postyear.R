rm(list = ls())

##functions
explanatory_aux <- function(tyear, lc_target_years, sub_cost_method="CONSTANT", seq_dist_method="LCS",cluster_method="PAM",n_cluster){
    
  yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  
  ##determine start year
  start_idx <- 2 + (which(yearls == tyear)-2)
  
  global_classes <- c('Barren',
                      'Water Bodies',
                      'Urban and Built-up Lands',
                      'Dense Forests',
                      'Open Forests',
                      'Farmland',
                      'Shrublands')
  
  global_shortclasses <- c('Ba', 
                           'W', 'Bu',
                           'DF', 'OF', 
                           'Fm',
                           'S')
  global_colors = c('#f9ffa4', 
                    '#1c0dff', '#fa0000', 
                    '#003f00', '#006c00', 
                    '#f096ff', 
                    '#dcd159') 
  
  global_features <- c(1,
                       1,1,
                       10,5,
                       3,
                       5)
  
  ## Load seq obj
  target_all_tb <- paste0(seqdata_dir,'/',tyear,'_target_tb')
  target_all_seq <- paste0(seqdata_dir,'/',tyear,'_target_seq')
  target_unique_tb <- paste0(seqdata_dir,'/',tyear,'_unique_tb')
  target_unique_seq <- paste0(seqdata_dir,'/',tyear,'_unique_seq')
  
  load(file=paste0(target_all_tb,".RSav"))
  load(file=paste0(target_all_seq,".RSav"))
  load(file=paste0(target_unique_tb,".RSav"))
  load(file=paste0(target_unique_seq,".RSav"))
  
  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)
  
  if(file.exists(paste0(seq_dist_path,".RSav"))){
    print("Loading seq dist file")
    load(file=paste0(seq_dist_path,".RSav"))
    seq_weights <- attr(seq_object,"weights")
  }
  
  target_cluster_tb <- paste0(clusters_dir,'/',cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name)
  load(file=paste0(target_cluster_tb,".RSav"))
  
  ##extract raster values
  ##add and prepare aux columns
  tab.target_geo = unique_seq
  coordinates(tab.target_geo) <- c("x", "y")
  mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))
  
  dem_val =raster::extract(dem, mypoints)
  slope_val =raster::extract(slope, mypoints)
  access_val =raster::extract(access, mypoints)
  
  tab.target_geo$dem = dem_val
  tab.target_geo$slope = slope_val
  tab.target_geo$access = access_val
  
  ## Create categorical data from covariates
  ## Two categories for distance and slope (binary variable)
  tab.target_geo$accessK2<-cut(tab.target_geo$access, c(0,500,max(tab.target_geo$access, na.rm = TRUE)))
  tab.target_geo$slopeK2<-cut(tab.target_geo$slope, c(-1,6,max(tab.target_geo$slope, na.rm = TRUE)))
  tab.target_geo$demK2<-cut(tab.target_geo$dem, c(0,250,max(tab.target_geo$dem, na.rm = TRUE)))
  
  # Compute and test the share of discrepancy explained by different categories on covariates 
  da1 <- dissassoc(seq_dist, weights=seq_weights,group = tab.target_geo$slopeK2, R = 50, weight.permutation="diss")
  print(da1$stat)
  da2 <- dissassoc(seq_dist, weights=seq_weights,group = tab.target_geo$demK2, R = 50, weight.permutation="diss")
  print(da2$stat)
  da3 <- dissassoc(seq_dist, weights=seq_weights,group = tab.target_geo$accessK2, R = 50, weight.permutation="diss")
  print(da3$stat)
  
  #relate
  ##add and prepare aux columns
  tab.target_geo = cluster_all
  coordinates(tab.target_geo) <- c("x", "y")
  mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))
  
  dem_val =raster::extract(dem, mypoints)
  slope_val =raster::extract(slope, mypoints)
  access_val =raster::extract(access, mypoints)
  
  tab.target_geo$dem = dem_val
  tab.target_geo$slope = slope_val
  tab.target_geo$access = access_val
  
  ## Create categorical data from covariates
  ## Two categories for distance and slope (binary variable)
  tab.target_geo$accessK2<-cut(tab.target_geo$access, c(0,500,max(tab.target_geo$access, na.rm = TRUE)))
  tab.target_geo$slopeK2<-cut(tab.target_geo$slope, c(-1,6,max(tab.target_geo$slope, na.rm = TRUE)))
  tab.target_geo$demK2<-cut(tab.target_geo$dem, c(0,250,max(tab.target_geo$dem, na.rm = TRUE)))
  
  # Compute and t
  tabe.seq <- seqecreate(seq_object_all, use.labels = FALSE)
  lc <- seqecontain(tabe.seq, event.list = c("DF"))
  lc_tab <- tab.target_geo[lc,]
  
  lc_tab_df = as.data.frame(lc_tab)
  
  classes <- c()
  for (year in start_idx:(length(yearls)+2)){
    classes <- unique(c(classes,lc_tab_df[,year]))
  }
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  lc.seq <- seqdef(lc_tab_df, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                       cpal = target_colors, labels = target_short)
  lc.seqe <- seqecreate(lc.seq, use.labels = FALSE)
  
  # Look for frequent event subsequences and plot the 10 most frequent ones.
  fsubseq <- seqefsub(lc.seqe, pmin.support = 0.05)
  # 10 Most common subsequences
  
  plot(fsubseq[1:10], col = "grey98")
  
  aux_vars<- c('demK2','slopeK2','accessK2','clusters')
  aux_names <- c('Elevation','Slope','Accessibility','Clusters')
  
  for (i in 1:length(aux_vars)){
    tcolumn <-which(names(lc_tab_df) == aux_vars[i])
    tname <- aux_names[i]
      
    discr1 <- seqecmpgroup(fsubseq, group = lc_tab_df[,tcolumn])
    
    file_path <- paste0(explanatory_dir,'/',tname,'_',cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name)
    
    if (tname != 'Clusters'){
      height = 400
      res_img = 70
    } else {
      height = 1200
      res_img = 100
    }
    
    png(file = paste0(file_path,'.png'), width = 560, height = height, res = res_img)
    plot(discr1[1:20],cex=1,cex.legend=1,legend.title=tname,cex.lab=0.8, cex.axis = 0.8)
    dev.off()
    
  }


}

# Working directory and libraries
library(TraMineR)
library(raster)
library(reshape2)
library(tidyr)
library(plyr)
library(rlist)
library(dplyr)
library(WeightedCluster)

##settings
tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distances")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_data")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clusters")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)
explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatory")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
proj <- CRS('+proj=longlat +ellps=WGS84')

dem <- raster(paste0(aux_dir,'/ancillary/gee/srtm.tif'))
slope <- raster(paste0(aux_dir,'/ancillary/gee/slope.tif'))
access <- raster(paste0(aux_dir,'/ancillary/gee/access.tif'))

##start###
targetyears = c(2005:2018)

cluster_method="WARD"
sub_cost_method = "TRATE"
seq_dist_method = "OM"
n_cluster=8

for (j in targetyears){
  tyear <- j
  print(tyear)
  explanatory_aux(tyear, lc_target_years, sub_cost_method=sub_cost_method,seq_dist_method=seq_dist_method,cluster_method=cluster_method,n_cluster=n_cluster)
}