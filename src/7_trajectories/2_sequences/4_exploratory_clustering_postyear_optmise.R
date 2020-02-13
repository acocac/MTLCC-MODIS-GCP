##TODO
# * best n clusters
# * 

rm(list = ls())

##functions
create_sequence_object <- function(target, tyear, start_idx, yearls, alphabet, target_short, target_colors, seq_dist_dir){
  
  # Create a TraMineR sequence object from csv file with alinged sequences
  
  # Parameters:
  # target: target
  # N: number of seqeunces
  # seed: seed for random sampling
  
  # Returns:
  # seq_object: A TraMineR sequence object with unique sequences and respective weights 
  agg_seq <- wcAggregateCases(target[,start_idx:(length(yearls)+2)])
  
  unique_seq <- target[agg_seq$aggIndex, ]
  
  seq_object <- seqdef(unique_seq, start_idx:(length(yearls)+2), weights = agg_seq$aggWeights, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)
  
  ## save target matrix
  seq_dist_name <- paste0(tyear,'_unique_tb')
  seq_dist_path <- paste0(seq_dist_dir,'/',seq_dist_name)
  if(!file.exists(paste0(seq_dist_path,".RSav"))){
    save(unique_seq, file=paste0(seq_dist_path,".RSav"))
  }
  
  ## save target seq
  seq_dist_name <- paste0(tyear,'_unique_seq')
  seq_dist_path <- paste0(seq_dist_dir,'/',seq_dist_name)
  if(!file.exists(paste0(seq_dist_path,".RSav"))){
    save(seq_object, file=paste0(seq_dist_path,".RSav"))
  }
  
  return (seq_object)
}

create_substitution_cost <- function(seq_object,method="CONSTANT",cval=2){
  
  # Create a substitution cost martix for computing distances between sequences
  
  # Parameters:
  # seq_object: the TraMineR sequence object
  # method: method for computing the substitution cost
  # CONSTANT: constant cost of cval
  # TRATE: using transistion rates from the data to compute costs (refer TraMineR documentation for formulation)
  
  # Returns:
  # seq_subcost: A TraMineR substitution cost object
  #   $sm : has the substitution cost matrix
  #   $indel : has the insertion and deletion costs
  
  seq_subcost <- seqcost(seq_object,method=method,cval=cval,weighted=FALSE)
  return (seq_subcost)
}

compute_sequence_distances <- function(seq_object,seq_subcost,method="LCS"){
  
  # Create a substitution cost martix for computing distances between sequences
  
  # Parameters:
  # seq_object: the TraMineR sequence object
  # seq_subcost: the substitution cost object
  # CONSTANT: constant cost of cval
  # TRATE: using transistion rates from the data to compute costs (refer TraMineR documentation for formulation)
  
  # Returns:
  # seq_distances: A lower triangular matrix with distances between sequences 
  
  seq_distances <- seqdist(seq_object,method = method,sm=seq_subcost$sm,full.matrix=FALSE)
  return (seq_distances)
}

evaluate_cluster_stats_medoids <- function(seq_dist,seq_wts,k_range){
  clus_stats <- wcKMedRange(seq_dist, weights=seq_wts, kvals=k_range)
  return(clus_stats)
}

evaluate_cluster_stats_hclust <- function(seq_dist,k_range,seq_weights){
  averageClust <- hclust(as.dist(seq_dist), method = "average", members=seq_weights)
  avgClustQual <- as.clustrange(averageClust, seq_dist, ncluster = max(k_range))
  return(avgClustQual)
}

save_sequence_distances <- function(seq_dist,seq_dist_path){
  # Save the distance matrix between sequences
  
  # Parameters:
  # seq_dist: the distance matrix
  # model_name: name of the file
  
  save(seq_dist, file=paste0(seq_dist_path,".RSav"))
}

find_optimal_clusters <- function(tyear, lc_target_years, seq_dist_dir, sub_cost_method="CONSTANT", seq_dist_method="LCS", bcluster="medoids", k_range){
# find_optimal_clusters <- function(tyear, lc_target_years, distances_dir){
    
  ##analysis
  infile <- paste0(indir,'/','resultsseq_tyear_',as.character(tyear),'_simple.rdata')
  
  results_df = rlist::list.load(infile)
  
  tab = results_df[4][[1]]
  tyear = results_df[3][[1]]
  
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
  
  classes <- c()
  for (year in start_idx:(length(yearls)+2)){
    classes <- unique(c(classes,tab[,year]))
  }
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  pos_year <- which(yearls == tyear)
  start_idx <- 2 + (which(yearls == tyear)-2)
  
  tab.seq <- seqdef(tab, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.target = tab.nonstable
  
  ## save target matrix
  seq_dist_name <- paste0(tyear,'_target_tb')
  seq_dist_path <- paste0(seq_dist_dir,'/',seq_dist_name)
  if(!file.exists(paste0(seq_dist_path,".RSav"))){
    save(tab.target, file=paste0(seq_dist_path,".RSav"))
  }
  
  seq_obj <- create_sequence_object(tab.target, tyear, start_idx, yearls, alphabet, target_short, target_colors, seq_dist_dir)
  seq_weights <- attr(seq_obj,"weights")
  
  ## Load distance matrix
  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(seq_dist_dir,'/',seq_dist_name)

  if(!file.exists(paste0(seq_dist_path,".RSav"))){
    ## Create substitution cost matrix
    if (sub_cost_method == "TRATE"){
      seq_subcost <- seqcost(seq_obj, method = "TRATE",with.missing = FALSE)
      seq_dist <- seqdist(seq_obj, method = "OM",indel = seq_subcost$indel, sm = seq_subcost$sm,with.missing = F)
    } else if(sub_cost_method == "FEATURES"){
      target_classes <- global_features[sort(classes)]
      print(target_classes)
      tab_state_features <- data.frame(state=target_classes)
      
      seq_subcost <- seqcost(seq_obj, method = "FEATURES",with.missing = FALSE, state.features = tab_state_features)
      seq_dist <- seqdist(seq_obj, method = "OM",indel = seq_subcost$indel, sm = seq_subcost$sm,with.missing = F)
    } else{
      seq_subcost <- create_substitution_cost(seq_obj,method=sub_cost_method,cval=2)
      seq_dist <- compute_sequence_distances(seq_obj,seq_subcost,method=seq_dist_method)
    }
    ## Save distance matrix
    save_sequence_distances(seq_dist,seq_dist_path)
  }else{
    print("Loading seq dist file")
    load(file=paste0(seq_dist_path,".RSav"))
  }

  if (bcluster == 'medoids'){
    cluster_stats <- evaluate_cluster_stats_medoids(seq_dist,seq_weights,k_range)
    df_statstics <- cluster_stats$stats
    clus_stats_name <- paste0("medoid_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(seq_dist_dir,"/",clus_stats_name,".csv"))

    png(file=paste0(seq_dist_dir,'/medoid_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_wcKM_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(x=k_range,y=df_statstics$ASW,type='l',col=2,main="ASW")
    plot(x=k_range,y=df_statstics$CH,type='l',col=2,main="CH")
    dev.off()
    
  } else{
    cluster_stats <- evaluate_cluster_stats_hclust(seq_dist,k_range,seq_weights)
    
    df_statstics <- cluster_stats$clustering
    clus_stats_name <- paste0("hclust_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(seq_dist_dir,"/",clus_stats_name,".csv"))
    
    png(file = paste0(seq_dist_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_zcore.png'), width = 560, height = 474, res = 99)
    plot(cluster_stats, norm = "zscore", withlegend = F)
    dev.off()

    png(file = paste0(seq_dist_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(cluster_stats$stats$ASW, type='l',col=2, withlegend = F, main="ASW")
    plot(cluster_stats$stats$CH, type='l',col=2, withlegend = F, main="CH")
    dev.off()
    
    summary(cluster_stats, max.rank = 2)
    
  }
  
}

cluster_sequences <- function(tyear, lc_target_years, seq_dist_dir, sub_cost_method="CONSTANT", seq_dist_method="LCS",cluster_method="PAM",n_cluster){
  
  # Master function to read csv file of aligned sequences, create sequence object, compute distance matrix and cluster them
  # Parameters:
  #   seq_file: path to aligned sequence
  #   N: sample size to analyse
  #   seed: random seed for sampling
  #   sub_cost_method: substirution cost method
  #   seq_dist_method: method for computing sequence distances
  #   n_cluster: number of clusters to compute
  
  # Returns:
  #   Saves the sequence distances and clustering results in the model path
  
  ## Load seq obj
  target_all_tb <- paste0(seq_dist_dir,'/',tyear,'_target_tb')
  target_unique_tb <- paste0(seq_dist_dir,'/',tyear,'_unique_tb')
  target_unique_seq <- paste0(seq_dist_dir,'/',tyear,'_unique_seq')
  
  load(file=paste0(target_all_tb,".RSav"))
  load(file=paste0(target_unique_tb,".RSav"))
  
  load(file=paste0(target_unique_seq,".RSav"))
  seq_weights <- attr(seq_object,"weights")

  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(seq_dist_dir,'/',seq_dist_name)

  if(file.exists(paste0(seq_dist_path,".RSav"))){
    print("Loading seq dist file")
    load(file=paste0(seq_dist_path,".RSav"))
    load(file=paste0(seq_dist_path,".RSav"))
    
  }else{
    print("Seq dist file does not exist")
  }
  
  if (cluster_method == 'PAM'){
  ## Cluster sequences
    clusters_pam <- pam_clustering(seq_dist,seq_weights,n_cluster,cluster_method)
    clusters_n <-  clusters_pam$clustering
  } else{
    ## Cluster based on OM transition rates
    clusters_ward <- hclust(as.dist(seq_dist),method="ward.D", members=seq_weights)
    # plot(clusterward)
    clusters_n <- cutree(clusters_ward, k = n_cluster)
  }
  
  unique_seq$clusters <- clusters_n
  
  ##merge clusters
  cluster_all <- tab.target %>% 
    left_join(select(unique_seq, clusters, sec), by = "sec")
  
  # Plot all the sequences within each cluster 
  graphics.off()
  png(file = paste0(distances_dir,"/",cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name,".png"), width = 500, height = 1200, res = 100)
  seqIplot(seq_object, group = unique_seq$clusters, sortv = "from.start", with.legend = F)
  dev.off()
  
}

pam_clustering <- function(seq_dist,seq_wts,k,cluster_method,seed=1729){
  # Clustering of the distance matrix using PAM/k-medoids alogirthm
  
  # Parameters:
  #   seq_dist: the distance matrix
  #   k: number of clusters
  # Returns:
  # cluster_pam: PAM cluster object which contains medoid IDs, cluster labels, etc.
  set.seed(seed)
  if (cluster_method=="PAM"){
    cluster_pam <- pam(seq_dist, k=k, diss = TRUE)
  }else{
    cluster_pam <- wcKMedoids(seq_dist,weights=seq_wts,k=k,cluster.only=TRUE,npass=5)
  }
  
  return (cluster_pam)
}

# Working directory and libraries
library(TraMineR)
library(raster)
library(reshape2)
library(sampling)
library(tidyr)
library(plyr)
library(rlist)
require(rasterVis)
library(maptools)
library(classInt)
library(dplyr)
library(WeightedCluster)


##settings
tile <- 'AMZ'
tyear <- 2004
lc_target_years <-c(2001,2019)

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts_postyear")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)
geodata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/geodata/postyear/output/clusters")
dir.create(geodata_dir, showWarnings = FALSE, recursive = T)
distances_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distances")
dir.create(distances_dir, showWarnings = FALSE, recursive = T)

# ##aux
# aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
# proj <- CRS('+proj=longlat +ellps=WGS84')
# 
# aoi_shp <- readShapeLines(paste0(aux_dir,'/aoi/amazon_raisg.shp'), proj4string=proj)
# dem <- raster(paste0(aux_dir,'/ancillary/gee/srtm.tif'))
# slope <- raster(paste0(aux_dir,'/ancillary/gee/slope.tif'))
# access <- raster(paste0(aux_dir,'/ancillary/gee/access.tif'))

##start###
targetyears = c(2004:2004)

bcluster <- 'hclust' #hclust
sub_cost_method = "TRATE"
seq_dist_method = "OM"

for (j in targetyears){
  tyear <- j
  find_optimal_clusters(tyear, lc_target_years, distances_dir, sub_cost_method=sub_cost_method,seq_dist_method=seq_dist_method,bcluster=bcluster,2:10)
}

##implement cluster##

##start###
targetyears = c(2004:2005)

cluster_method="WARD"
sub_cost_method = "TRATE"
seq_dist_method = "OM"
n_cluster=5

for (j in targetyears){
  tyear <- j
  cluster_sequences(tyear, lc_target_years, distances_dir, sub_cost_method=sub_cost_method,seq_dist_method=seq_dist_method,cluster_method=cluster_method,n_cluster)
}


## Load cluster
sub_cost_method = "TRATE"
seq_dist_method = "OM"
ncluster=4

for (j in targetyears){
  ## Load matrix and seq
  tyear <- j
  
  target_all_tb <- paste0(distances_dir,'/',tyear,'_target_tb')
  target_unique_tb <- paste0(distances_dir,'/',tyear,'_unique_tb')
  target_unique_seq <- paste0(distances_dir,'/',tyear,'_unique_seq')
  
  load(file=paste0(target_all_tb,".RSav"))
  load(file=paste0(target_unique_tb,".RSav"))
  load(file=paste0(target_unique_seq,".RSav"))
  
  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(distances_dir,'/',seq_dist_name)
  load(file=paste0(seq_dist_path,".RSav"))
  
  ## Cluster based on OM transition rates
  clusterward <- hclust(as.dist(seq_dist),method="ward.D")
  # plot(clusterward)
  cl_cut <- cutree(clusterward, k = ncluster)
  
  unique_seq$clusters <- cl_cut

  ##merge clusters
  cluster_all <- tab.target %>% 
    left_join(select(unique_seq, clusters, sec), by = "sec")
  
  # Plot all the sequences within each cluster 
  graphics.off()
  png(file = paste0(distances_dir,"/clustersk",as.character(ncluster),"_",seq_dist_name,".png"), width = 500, height = 1200, res = 100)
  seqIplot(seq_object, group = unique_seq$clusters, sortv = "from.start", with.legend = F)
  dev.off()
  
  # Elaborate raster OM1
  xyz <- as.data.frame(cbind(cluster_all$x,cluster_all$y,cluster_all$clusters))
  names(xyz) <- c("x","y","z")
  xyz <- xyz[complete.cases(xyz), ]
  coordinates(xyz) <- ~ x + y
  gridded(xyz) <- TRUE
  raster_com1 <- raster(xyz)
  crs(raster_com1) <- CRS('+init=EPSG:4326')
  
  writeRaster(raster_com1,paste0(geodata_dir,"/clustersk",as.character(ncluster),"_",seq_dist_name,".tif"), format="GTiff", overwrite=TRUE)
  
}

ssplot(seq_object, type='I')


require(seqHMM)


##analysis with simple classes

##DHD
diss_target <- seqdist(target_seq_unique, method = "DHD")

graphics.off()
averageClust <- hclust(as.dist(diss_target), method = "average")
avgClustQual <- as.clustrange(averageClust, diss_target, ncluster = 10)
png(file = paste0(chart_dir,"/clustersDHD","_def",tyear,".png"), width = 560, height = 474, res = 99)
plot(avgClustQual, norm = "zscore", withlegend = F)
dev.off()
summary(avgClustQual, max.rank = 2)

##OM - TR
costs.tr <- seqcost(target_seq_unique, method = "TRATE",with.missing = FALSE)
print(costs.tr)
dist.om1 <- seqdist(target_seq_unique, method = "OM",indel = costs.tr$indel, sm = costs.tr$sm,with.missing = F)

graphics.off()
averageClust <- hclust(as.dist(dist.om1), method = "average")
avgClustQual <- as.clustrange(averageClust, dist.om1, ncluster = 10)
png(file = paste0(chart_dir,"/clusters_OM1","_def",tyear,".png"), width = 560, height = 474, res = 99)
plot(avgClustQual, norm = "zscore", withlegend = F)
dev.off()
summary(avgClustQual, max.rank = 2)

##averagetree - only work in pc with admin rights
averageTree <- as.seqtree(averageClust, seqdata=target_seq_unique, diss=dist.om1, ncluster=6)
## Graphical representation of the tree (you need to have Graphviz installed before lauchning R)
seqtreedisplay(averageTree, type="d", border=NA,  showdepth=TRUE) ##only working on laptop

## Compute PAM clustering and cluster quality measure for different number of groups (ranging from 2 to 10)
pamClustRange <- wcKMedRange(dist.om1, kvals=2:10, weights=aggtarget$aggWeights)

## Print the 2 best number of group according to each quality measure for the PAM clustering
summary(pamClustRange, max.rank=2)

## The best clustering was found using average clustering in 5 groups according to ASW (average silhouette width)
seqdplot(target_seq_unique, group=avgClustQual$clustering$cluster5, border=NA)

## Clustering was made on distinct sequences
## Recover the clustering solution in the original (full) dataset
uniqueCluster5 <- avgClustQual$clustering$cluster5
tab.target$cluster5 <- uniqueCluster5[aggtarget$disaggIndex]


#######################################################################################
# Compute distances between sequences using different dissimilarity indices

## OM with substitution costs based on transition
## probabilities and indel set as half the maximum
## substitution cost
costs.tr <- seqcost(target_seq_unique, method = "TRATE",with.missing = FALSE)
print(costs.tr)
dist.om1 <- seqdist(target_seq_unique, method = "OM",indel = costs.tr$indel, sm = costs.tr$sm,with.missing = F)
dim(dist.om1)

## Cluster based on OM transition rates
clusterward_om1 <- hclust(as.dist(dist.om1),method="ward.D")
plot(clusterward_om1)
cl_om1 <- cutree(clusterward_om1, k = 5)
uniquetarget$clusterom1 <- cl_om1
head(tab)

##merge clusters
cluster_all <- tab.target %>% 
  left_join(select(uniquetarget, clusterom1, sec), by = "sec")


# Plot all the sequences within each cluster para los 4 métodos
# OM1
graphics.off()
png(file = paste0(chart_dir,"/clusters_OM1_seq_def",tyear,".png"), width = 600, height = 400, res = 150)
seqIplot(target_seq_unique, group = uniquetarget$clusterom1, sortv = "from.start", with.legend = F)
dev.off()

######### Plot clusters' spatial distribution

# Elaborate raster OM1
xyz <- as.data.frame(cbind(cluster_all$x,cluster_all$y,cluster_all$clusterom1))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_com1 <- raster(xyz)
plot(raster_com1)
crs(raster_com1) <- CRS('+init=EPSG:4326')

writeRaster(raster_com1,paste0(geodata_dir,"/clusterOM1_",tyear,".tif"), format="GTiff", overwrite=TRUE)

#######################################################################################
## Run discrepancy analyses to study how sequences are related to covariates

# Compute and test the share of discrepancy explained by different categories on covariates 
da1 <- dissassoc(dist.om1, group = uniquetarget$slopeK, R = 50)
print(da1$stat)
da2 <- dissassoc(dist.om1, group = uniquetarget$demK, R = 50)
print(da2$stat)
da3 <- dissassoc(dist.om1, group = uniquetarget$accessK2, R = 50)
print(da3$stat)


# Selecting event subsequences:
# The analysis was restricted to sequences that exhibit the state Mosaic

tabe.seq <- seqecreate(tab.target.seq, use.labels = FALSE)
mosaic <- seqecontain(tabe.seq, event.list = c("DF"))
mosaic_tab <- cluster_all[mosaic,]
mosaic.seq <- seqdef(mosaic_tab, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                                cpal = target_colors, labels = target_short)
mosaic.seqe <- seqecreate(mosaic.seq, use.labels = FALSE)

# Look for frequent event subsequences and plot the 10 most frequent ones.
fsubseq <- seqefsub(mosaic.seqe, pmin.support = 0.05)
head(fsubseq)
# 10 Most common subsequences
plot(fsubseq[1:10], col = "grey98")

# Determine the subsequences of transitions which best discriminate the groups as
# areas close and faraway from roads
discr1 <- seqecmpgroup(fsubseq, group = mosaic_tab$demK)
plot(discr1[1:10],cex=1,cex.legend=1,legend.title="Elevation",cex.lab=0.8, cex.axis = 0.8)
# areas with moderate vs steep slope
discr2 <- seqecmpgroup(fsubseq, group = mosaic_tab$slopeK2)
plot(discr2[1:10],cex=1,cex.legend=1,legend.title="Slope",cex.lab=0.8, cex.axis = 0.8)
# areas with accesability
discr3 <- seqecmpgroup(fsubseq, group = mosaic_tab$accessK2)
plot(discr3[1:10],cex=1,cex.legend=1,legend.title="Accessibility",cex.lab=0.8, cex.axis = 0.8)
# clusters of sequences
discr4 <- seqecmpgroup(fsubseq, group = mosaic_tab$clusterom1)
plot(discr4[1:10],cex=1,cex.legend=1,legend.title="Clusters OM1",cex.lab=0.8, cex.axis = 0.8)
