##TODO
# * best n clusters
# * 

rm(list = ls())

##functions
create_sequence_object <- function(target, tyear, start_idx, yearls, alphabet, target_short, target_colors){
  
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
  file_name <- paste0(tyear,'_unique_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(unique_seq, file=paste0(file_path,".RSav"))
  }
  
  ## save target seq
  file_name <- paste0(tyear,'_unique_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(seq_object, file=paste0(file_path,".RSav"))
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
  avgClustQual <- as.clustrange(averageClust, diss=seq_dist, weights=seq_weights,  ncluster = max(k_range))
  return(avgClustQual)
}

save_sequence_distances <- function(seq_dist,seq_dist_path){
  # Save the distance matrix between sequences
  
  # Parameters:
  # seq_dist: the distance matrix
  # model_name: name of the file
  
  save(seq_dist, file=paste0(seq_dist_path,".RSav"))
}

find_optimal_clusters <- function(tyear, lc_target_years, sub_cost_method="CONSTANT", seq_dist_method="LCS", bcluster="medoids", k_range){
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
                    '#db00ff', 
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
  
  tab.seq <- seqdef(tab, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.target = tab.nonstable
  
  seq_object_all <- seqdef(tab.target, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                              cpal = target_colors, labels = target_short)
  
  ## save seq all
  file_name <- paste0(tyear,'_target_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)

  if(!file.exists(paste0(file_path,".RSav"))){
    save(seq_object_all, file=paste0(file_path,".RSav"))
  }
  
  ## save target matrix
  file_name <- paste0(tyear,'_target_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(tab.target, file=paste0(file_path,".RSav"))
  }
  
  seq_obj <- create_sequence_object(tab.target, tyear, start_idx, yearls, alphabet, target_short, target_colors)
  seq_weights <- attr(seq_obj,"weights")
  
  ## Load distance matrix
  file_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  file_path <- paste0(seqdist_dir,'/',file_name)

  if(!file.exists(paste0(file_path,".RSav"))){
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
    save_sequence_distances(seq_dist,file_path)
  }else{
    print("Loading seq dist file")
    load(file=paste0(file_path,".RSav"))
  }

  if (bcluster == 'medoids'){
    cluster_stats <- evaluate_cluster_stats_medoids(seq_dist,seq_weights,k_range)
    df_statstics <- cluster_stats$stats
    clus_stats_name <- paste0("medoid_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(clusters_dir,"/",clus_stats_name,".csv"))

    png(file=paste0(clusters_dir,'/medoid_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_wcKM_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(x=k_range,y=df_statstics$ASW,type='l',col=2,main="ASW")
    plot(x=k_range,y=df_statstics$CH,type='l',col=2,main="CH")
    dev.off()
    
  } else{
    cluster_stats <- evaluate_cluster_stats_hclust(seq_dist,k_range,seq_weights)
    
    df_statstics <- cluster_stats$clustering
    clus_stats_name <- paste0("hclust_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(clusters_dir,"/",clus_stats_name,".csv"))
    
    png(file = paste0(clusters_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_zcore.png'), width = 560, height = 474, res = 99)
    plot(cluster_stats, norm = "zscore", withlegend = F)
    dev.off()

    png(file = paste0(clusters_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(cluster_stats$stats$ASW, type='l',col=2, withlegend = F, main="ASW")
    plot(cluster_stats$stats$CH, type='l',col=2, withlegend = F, main="CH")
    dev.off()
    
    summary(cluster_stats, max.rank = 2)
    
  }
  
}

cluster_sequences <- function(tyear, lc_target_years, sub_cost_method="CONSTANT", seq_dist_method="LCS",cluster_method="PAM",n_cluster, writeraster_cluster=FALSE){
  
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
  target_all_tb <- paste0(seqdata_dir,'/',tyear,'_target_tb')
  target_unique_tb <- paste0(seqdata_dir,'/',tyear,'_unique_tb')
  target_unique_seq <- paste0(seqdata_dir,'/',tyear,'_unique_seq')
  
  load(file=paste0(target_all_tb,".RSav"))
  load(file=paste0(target_unique_tb,".RSav"))
  
  load(file=paste0(target_unique_seq,".RSav"))
  seq_weights <- attr(seq_object,"weights")

  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)

  if(file.exists(paste0(seq_dist_path,".RSav"))){
    print("Loading seq dist file")
    load(file=paste0(seq_dist_path,".RSav"))
  }else{
    print("Seq dist file does not exist")
  }
  
  if (cluster_method == 'PAM'){
  ## Cluster sequences
    clusters_out <- pam_clustering(seq_dist,seq_weights,n_cluster,cluster_method)
    clusters_n <-  clusters_out$clustering
  } else{
    ## Cluster based on OM transition rates
    clusters_out <- hclust(as.dist(seq_dist),method="ward.D", members=seq_weights)
    # plot(clusterward)
    clusters_n <- cutree(clusters_out, k = n_cluster)
  }
  
  unique_seq$clusters <- clusters_n
  
  ##merge clusters
  cluster_all <- tab.target %>% 
    left_join(select(unique_seq, clusters, sec), by = "sec")
  
  # Plot all the sequences within each cluster 
  graphics.off()
  png(file = paste0(clusters_dir,"/",cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name,".png"), width = 500, height = 1200, res = 100)
  seqIplot(seq_object, group = unique_seq$clusters, sortv = "from.start", with.legend = F)
  dev.off()
  
  ## save target seq
  file_path <- paste0(clusters_dir,'/',cluster_method,"_dendogram_",seq_dist_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(clusters_out, file=paste0(file_path,".RSav"))
  }
  
  file_name <- paste0(tyear,'_target_clusters')
  file_path <- paste0(clusters_dir,'/',cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(cluster_all, file=paste0(file_path,".RSav"))
  }
  
  if (writeraster_cluster == TRUE){
    # Elaborate raster OM1
    xyz <- as.data.frame(cbind(cluster_all$x,cluster_all$y,cluster_all$clusters))
    names(xyz) <- c("x","y","z")
    xyz <- xyz[complete.cases(xyz), ]
    coordinates(xyz) <- ~ x + y
    gridded(xyz) <- TRUE
    raster_cluster <- raster(xyz)
    crs(raster_cluster) <- CRS('+init=EPSG:4326')
    
    file_path <- paste0(clusters_dir,'/',cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name)
    writeRaster(raster_cluster,paste0(file_path,".tif"), format="GTiff", overwrite=TRUE)
  }
  
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

find_optimal_clusters <- function(tyear, lc_target_years, sub_cost_method="CONSTANT", seq_dist_method="LCS", bcluster="medoids", k_range){
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
  
  start_idx <- 2 + (which(yearls == tyear)-2)
  
  tab.seq <- seqdef(tab, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.target = tab.nonstable
  
  seq_object_all <- seqdef(tab.target, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                           cpal = target_colors, labels = target_short)
  
  ## save seq all
  file_name <- paste0(tyear,'_target_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)
  
  if(!file.exists(paste0(file_path,".RSav"))){
    save(seq_object_all, file=paste0(file_path,".RSav"))
  }
  
  ## save target matrix
  file_name <- paste0(tyear,'_target_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(tab.target, file=paste0(file_path,".RSav"))
  }
  
  seq_obj <- create_sequence_object(tab.target, tyear, start_idx, yearls, alphabet, target_short, target_colors)
  seq_weights <- attr(seq_obj,"weights")
  
  ## Load distance matrix
  file_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  file_path <- paste0(seqdist_dir,'/',file_name)
  
  if(!file.exists(paste0(file_path,".RSav"))){
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
    save_sequence_distances(seq_dist,file_path)
  }else{
    print("Loading seq dist file")
    load(file=paste0(file_path,".RSav"))
  }
  
  if (bcluster == 'medoids'){
    cluster_stats <- evaluate_cluster_stats_medoids(seq_dist,seq_weights,k_range)
    df_statstics <- cluster_stats$stats
    clus_stats_name <- paste0("medoid_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(clusters_dir,"/",clus_stats_name,".csv"))
    
    png(file=paste0(clusters_dir,'/medoid_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_wcKM_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(x=k_range,y=df_statstics$ASW,type='l',col=2,main="ASW")
    plot(x=k_range,y=df_statstics$CH,type='l',col=2,main="CH")
    dev.off()
    
  } else{
    cluster_stats <- evaluate_cluster_stats_hclust(seq_dist,k_range,seq_weights)
    
    df_statstics <- cluster_stats$clustering
    clus_stats_name <- paste0("hclust_cluster_stats_",tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range))
    write.csv(df_statstics,file=paste0(clusters_dir,"/",clus_stats_name,".csv"))
    
    png(file = paste0(clusters_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_zcore.png'), width = 560, height = 474, res = 99)
    plot(cluster_stats, norm = "zscore", withlegend = F)
    dev.off()
    
    png(file = paste0(clusters_dir,'/hclust_cluster_statistics_',tyear,'_',sub_cost_method,'_',seq_dist_method,'_k_',min(k_range),'_to_',max(k_range),'_ASW_CH.png'), width = 560, height = 474, res = 99)
    par(mfrow=c(1:2))
    plot(cluster_stats$stats$ASW, type='l',col=2, withlegend = F, main="ASW")
    plot(cluster_stats$stats$CH, type='l',col=2, withlegend = F, main="CH")
    dev.off()
    
    summary(cluster_stats, max.rank = 2)
    
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
tyear <- 2004
lc_target_years <-c(2001,2019)

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
geodata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/geodata/postyear/output/clusters")
dir.create(geodata_dir, showWarnings = FALSE, recursive = T)
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distances")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_data")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clusters")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)

##start###
targetyears = c(2004:2004)

bcluster <- 'hclust' #hclust
sub_cost_method = "TRATE"
seq_dist_method = "OM"

for (j in targetyears){
  tyear <- j
  find_optimal_clusters(tyear, lc_target_years, sub_cost_method=sub_cost_method,seq_dist_method=seq_dist_method,bcluster=bcluster,2:10)
}

##implement cluster##
targetyears = c(2004:2004)

cluster_method="WARD"
sub_cost_method = "TRATE"
seq_dist_method = "OM"
n_cluster=6

for (j in targetyears){
  tyear <- j
  cluster_sequences(tyear, lc_target_years, sub_cost_method=sub_cost_method,seq_dist_method=seq_dist_method,cluster_method=cluster_method,n_cluster=n_cluster, writeraster_cluster=TRUE)
}