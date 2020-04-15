rm(list = ls())

create_sequence_object <- function(target_period, maxseq){
  # 
  # fn <- paste0(target_period,'_unique_tb.RSav')
  # load(paste0(seqdata_dir,'/',fn))
  # 
  # fn <- paste0(target_period,'_unique_seq.RSav')
  # load(paste0(seqdata_dir,'/',fn))
  # seq_weights <- attr(seq_object,"weights")
  
  fn <- paste0(target_period,'_target_tb.RSav')
  load(paste0(seqdata_dir,'/',fn))
  
  target_tb_subset_all = target_tb[,0:(maxseq+2)]
  target_tb_subset_all <- target_tb_subset_all[complete.cases(target_tb_subset_all),]
  
  print(paste0('Dimensions ori data: ', dim(target_tb_subset_all)[1]))
  
  #create stratas
  std.df <- apply(target_tb_subset_all[,3:dim(target_tb_subset_all)[2]], 1, sd) 
  
  target_tb_subset = target_tb_subset_all[std.df!=0,]
  
  print(paste0('Dimensions analysed data: ', dim(target_tb_subset)[1]))
  
  tyear = paste0(target_period,'_minperiod',maxseq)
  
  agg_seq_subset <- wcAggregateCases(target_tb_subset[,3:dim(target_tb_subset)[2]])
  
  unique_seq_subset <- target_tb_subset[agg_seq_subset$aggIndex, ]
  
  classes = unique(as.vector(as.matrix(unique_seq_subset[,3:dim(unique_seq_subset)[2]])))
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  seq_obj <- seqdef(unique_seq_subset, 3:dim(unique_seq_subset)[2], weights = agg_seq_subset$aggWeights, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)
  
  ## save target matrix
  file_name <- paste0(tyear,'_target_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(target_tb_subset, file=paste0(file_path,".RSav"))
  }
  
  ## save target matrix
  file_name <- paste0(tyear,'_unique_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(unique_seq_subset, file=paste0(file_path,".RSav"))
  }
  
  ## save target seq
  file_name <- paste0(tyear,'_unique_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(seq_obj, file=paste0(file_path,".RSav"))
  }
  
  return(seq_obj)
  
}

create_sequence_object_maxseq <- function(target_period, maxseq){
  # 
  # fn <- paste0(target_period,'_unique_tb.RSav')
  # load(paste0(seqdata_dir,'/',fn))
  # 
  # fn <- paste0(target_period,'_unique_seq.RSav')
  # load(paste0(seqdata_dir,'/',fn))
  # seq_weights <- attr(seq_object,"weights")
  
  fn <- paste0(target_period,'_target_tb.RSav')
  load(paste0(seqdata_dir,'/',fn))
  
  target_tb_subset_all = target_tb[,0:(maxseq+2)]
  target_tb_subset_all <- target_tb_subset_all[complete.cases(target_tb_subset_all),]
  
  print(paste0('Dimensions ori data: ', dim(target_tb_subset_all)[1]))
  
  #create stratas
  std.df <- apply(target_tb_subset_all[,3:dim(target_tb_subset_all)[2]], 1, sd) 
  
  target_tb_subset = target_tb_subset_all[std.df!=0,]
  
  print(paste0('Dimensions analysed data: ', dim(target_tb_subset)[1]))
  
  tyear = paste0(target_period,'_minperiod',maxseq)
  
  agg_seq_subset <- wcAggregateCases(target_tb_subset[,3:dim(target_tb_subset)[2]])
  
  unique_seq_subset <- target_tb_subset[agg_seq_subset$aggIndex, ]
  
  classes = unique(as.vector(as.matrix(unique_seq_subset[,3:dim(unique_seq_subset)[2]])))
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  seq_obj <- seqdef(unique_seq_subset, 3:dim(unique_seq_subset)[2], weights = agg_seq_subset$aggWeights, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)
  
  ## save target matrix
  file_name <- paste0(tyear,'_target_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(target_tb_subset, file=paste0(file_path,".RSav"))
  }
  
  ## save target matrix
  file_name <- paste0(tyear,'_unique_tb')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(unique_seq_subset, file=paste0(file_path,".RSav"))
  }
  
  ## save target seq
  file_name <- paste0(tyear,'_unique_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(seq_obj, file=paste0(file_path,".RSav"))
  }
  
  return(seq_obj)
  
}

save_sequence_distances <- function(seq_dist,seq_dist_path){
  # Save the distance matrix between sequences
  
  # Parameters:
  # seq_dist: the distance matrix
  # model_name: name of the file
  
  save(seq_dist, file=paste0(seq_dist_path,".RSav"))
}

eval_maxseq <- function(target_period, maxseq, n_cluster){
  
  tyear = paste0(target_period,'_minperiod',maxseq)
  
  file_name <- paste0(tyear,'_unique_seq')
  file_path <- paste0(seqdata_dir,'/',file_name)
  
  if(file.exists(paste0(file_path,".RSav"))){
    print("Loading seq file")
    load(file=paste0(file_path,".RSav"))
  }else{
    print("Seq file does not exist")
    seq_obj <- create_sequence_object_maxseq(target_period, maxseq)
  }
  
  seq_weights <- attr(seq_obj,"weights")
  
  seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
  seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)
  
  if(file.exists(paste0(seq_dist_path,".RSav"))){
    print("Loading seq dist file")
    load(file=paste0(seq_dist_path,".RSav"))
  }else{
    print("Seq dist file does not exist")
    seq_subcost <- seqcost(seq_obj, method = "TRATE",with.missing = FALSE)
    seq_dist <- seqdist(seq_obj, method = "OM",indel = seq_subcost$indel, sm = seq_subcost$sm,with.missing = F)
    save_sequence_distances(seq_dist,seq_dist_path)
  }
  
  ## save target seq
  averageClust <- hclust(as.dist(seq_dist), method = "ward.D", members=seq_weights)
  
  file_path <- paste0(clusters_dir,"/heatmap_",cluster_method,"_",seq_dist_name)
  if(!file.exists(paste0(file_path,".png"))){
    graphics.off()
    png(file = paste0(clusters_dir,"/heatmap_",cluster_method,"_",seq_dist_name,".png"), width = 850, height = 850, res = 150)
    seq_heatmap(seq_obj, averageClust, labCol=1:maxseq)
    dev.off()
  }
    
  seqtree_obj <- as.seqtree(averageClust, seqdata = seq_obj, diss = seq_dist, ncluster = n_cluster)
  
  file_path <- paste0(clusters_dir,'/',cluster_method,"_tree",n_cluster,"_",seq_dist_name)
  if(!file.exists(paste0(file_path,".RSav"))){
    save(seqtree_obj, file=paste0(file_path,".RSav"))
  }

  clus_stability_name <- paste0(cluster_method,"_clustersk",as.character(n_cluster),"_",seq_dist_name,"_stability_cluster")
  
  if(file.exists(paste0(clus_stability_name,".RSav"))){
    load(paste0(clus_stability_name,".RSav"))
    print(clusters_stability)
    
  } else{
    clusters_stability <- clusterboot(seq_dist, B=10, distances = TRUE, clustermethod = disthclustCBI, method =
                                        "ward.D", k = n_cluster)
    
    file_path <- paste0(clusters_dir,"/",clus_stability_name)
    if(!file.exists(paste0(file_path,".RSav"))){
      save(clusters_stability, file=paste0(file_path,".RSav"))
    }
  }
  
  return(clusters_stability)
  
}
  
library(data.table)
library(TraMineR, quietly = TRUE)
library(JLutils, quietly = TRUE)
library(future.apply)
library(hrbrthemes)
library(viridis)

##settings
tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distancesv2")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_datav2")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clustersv2")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)
charts_dir <- paste0("E:/acocac/research/",tile,"/trajectories/chartsv2")
dir.create(charts_dir, showWarnings = FALSE, recursive = T)

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

####SA - Minimum period####
####all period###
targetyears = c(2004:2016)
tyear = paste0(min(targetyears),'-',max(targetyears))  

file_name <- paste0(min(targetyears),'-',max(targetyears),'_target_tb')
load(paste0(seqdata_dir,'/',file_name,'.RSav'))

classes = unique(na.omit(as.vector(as.matrix(target_tb[,3:(dim(target_tb)[2]-1)]))))

target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

file_seq <- paste0(min(targetyears),'-',max(targetyears),'_unique_seq')
seq_data_path <- paste0(seqdata_dir,'/',file_seq)

if(file.exists(paste0(seq_data_path,".RSav"))){
  print("Loading seq file")
  load(file=paste0(seq_data_path,".RSav"))
  seq_weights <- attr(seq_obj,"weights")
}else{
  print("Seq file does not exist")
  seq_object <- create_sequence_object(target_tb, tyear, alphabet, target_short, target_colors)
}

seq_weights <- attr(seq_object,"weights")

###distances
sub_cost_method="TRATE"
seq_dist_method="OM"
cluster_method="WARD"

seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)

if(file.exists(paste0(seq_dist_path,".RSav"))){
  print("Loading seq dist file")
  load(file=paste0(seq_dist_path,".RSav"))
}else{
  print("Seq dist file does not exist")
  seq_subcost <- seqcost(seq_obj, method = "TRATE",with.missing = FALSE)
  seq_dist <- seqdist(seq_obj, method = "OM",indel = seq_subcost$indel, sm = seq_subcost$sm,with.missing = F)
  save_sequence_distances(seq_dist,seq_dist_path)
}

###clustering
seq_clust_name <- paste0(cluster_method,"_dendogram_",tyear,'_',sub_cost_method,'_',seq_dist_method)
seq_clust_path <- paste0(clusters_dir,'/',seq_clust_name)

if(file.exists(paste0(seq_clust_path,".RSav"))){
  print("Loading cluster file")
  load(file=paste0(seq_clust_path,".RSav"))
}else{
  print("Cluster file does not exist")
  averageClust <- hclust(as.dist(seq_dist), method = "ward.D", members=seq_weights)
  save(averageClust, file=paste0(seq_clust_path,".RSav"))
}

graphics.off()
png(file = paste0(clusters_dir,"/heatmap_",cluster_method,"_",seq_dist_name,".png"), width = 850, height = 850, res = 150)
seq_heatmap(seq_object, averageClust, labCol=1:18)
dev.off()

####### asssess min period ##### 
plan(multiprocess, workers = 13) ## Parallelize using four cores
set.seed(123)

sub_cost_method="TRATE"
seq_dist_method="OM"
cluster_method="WARD"

targetyears = c(2004:2016)
target_period = paste0(min(targetyears),'-',max(targetyears))  

maxseqs = seq(6,16,2)
clusters = c(2:10)

exp.df <- expand.grid(maxseqs, clusters)

cs_run <- function(i) {
  clusters_stability = eval_maxseq(target_period, exp.df[i,1], exp.df[i,2])
  return(transpose(as.data.frame(clusters_stability$bootmean)))
}

clusterStability_all = future.apply::future_lapply(1:nrow(exp.df), FUN = cs_run, future.seed = TRUE)

results_tb <- rbindlist(clusterStability_all, fill=TRUE)
results_tb$mean = rowMeans(results_tb, na.rm = T)
results_tb$sd = matrixStats::rowSds(as.matrix(results_tb), na.rm = T)

##merge 
final_tb <- cbind(exp.df, results_tb)

## final plot
png(file = paste0(charts_dir,"/jaccard.png"), width = 850, height = 850, res = 150)
ggplot2::ggplot(final_tb, aes(x=Var1, y=Var2, size=mean)) +
  geom_point(shape=21, color="black", fill = "black") +
  scale_size(name="", range=c(0,10))  +
  theme_ipsum_rc(axis_title_size = 18, base_size=17) +
  theme(legend.position="bottom") +
  labs(x="Minimum period", y="Number of clusters",
       title="Jaccard Bootstrap Mean (JBM)") +
  theme(legend.text=element_text(size=16)) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank())
dev.off()

  #theme(legend.position = "none")
