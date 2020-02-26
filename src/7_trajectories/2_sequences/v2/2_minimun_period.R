rm(list = ls())

create_sequence_object <- function(target, tyear, alphabet, target_short, target_colors){
  
  # Create a TraMineR sequence object from csv file with alinged sequences
  
  # Parameters:
  # target: target
  # N: number of seqeunces
  # seed: seed for random sampling
  
  # Returns:
  # seq_object: A TraMineR sequence object with unique sequences and respective weights 
  agg_seq <- wcAggregateCases(target[,3:(dim(target)[2]-1)])
  
  unique_seq <- target[agg_seq$aggIndex, ]
  
  seq_object <- seqdef(unique_seq, 3:(dim(target)[2]-1), weights = agg_seq$aggWeights, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)
  
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

save_sequence_distances <- function(seq_dist,seq_dist_path){
  # Save the distance matrix between sequences
  
  # Parameters:
  # seq_dist: the distance matrix
  # model_name: name of the file
  
  save(seq_dist, file=paste0(seq_dist_path,".RSav"))
}

library(data.table)
library(TraMineR, quietly = TRUE)
library(JLutils, quietly = TRUE)

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
targetyears = c(2004:2016)
target_period = paste0(min(targetyears),'-',max(targetyears))  

maxseq = 12
n_cluster=7

sub_cost_method="TRATE"
seq_dist_method="OM"
cluster_method="WARD"

fn <- paste0(target_period,'_unique_tb')
load(paste0(seqdata_dir,'/',fn))

fn <- paste0(target_period,'_unique_seq')
load(paste0(seqdata_dir,'/',fn))
seq_weights <- attr(seq_object,"weights")

tyear = paste0(target_period,'_minperiod',maxseq)

unique_seq$seq_length <- seqlength(seq_object)
unique_seq_subset <- unique_seq[seq_length >= maxseq, 0:(maxseq+2)]

idx_subset <- which(unique_seq$seq_length >= maxseq)
seq_weights_subset <- seq_weights[idx_subset]

classes = unique(as.vector(as.matrix(unique_seq_subset[,3:(maxseq+2)])))

target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

seq_object_subset <- seqdef(unique_seq_subset, 3:(dim(unique_seq_subset)[2]), weights = seq_weights_subset, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)

seq_subcost_subset <- seqcost(seq_object_subset, method = "TRATE",with.missing = FALSE)
seq_dist_subset <- seqdist(seq_object_subset, method = "OM",indel = seq_subcost_subset$indel, sm = seq_subcost_subset$sm,with.missing = F)

seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)

save_sequence_distances(seq_dist_subset,seq_dist_path)

averageClust_subset <- hclust(as.dist(seq_dist_subset), method = "ward.D", members=seq_weights_subset)

#data.hc <- clusterboot(seq_dist_subset, B=10, distances = TRUE, clustermethod = disthclustCBI, method =
#                         "ward.D", k = n_cluster)

## save target seq
file_path <- paste0(clusters_dir,'/',cluster_method,"_dendogram_",seq_dist_name)
if(!file.exists(paste0(file_path,".RSav"))){
  save(averageClust_subset, file=paste0(file_path,".RSav"))
}

graphics.off()
png(file = paste0(clusters_dir,"/heatmap_",cluster_method,"_",seq_dist_name,".png"), width = 850, height = 850, res = 150)
seq_heatmap(seq_object_subset, averageClust_subset, labCol=1:maxseq)
dev.off()

seqtree_subset <- as.seqtree(averageClust_subset, seqdata = seq_object_subset, diss = seq_dist_subset, ncluster = n_cluster)

file_path <- paste0(clusters_dir,'/',cluster_method,"_tree",n_cluster,"_",seq_dist_name)
if(!file.exists(paste0(file_path,".RSav"))){
  save(seqtree_subset, file=paste0(file_path,".RSav"))
}