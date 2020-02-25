rm(list = ls())

subset_target <- function(tyear, lc_target_years){
  
  ##analysis
  infile <- paste0(indir,'/','resultsseq_tyear_',as.character(tyear),'_simple.rdata')
  
  results_df = rlist::list.load(infile)
  
  tab = results_df[4][[1]]
  tyear = results_df[3][[1]]
  
  yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  
  ##determine start/end year
  start_idx <- 2 + (which(yearls == tyear)-2)
  end_idx <- 2 + (which(yearls == tyear)+2)
  
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
  
  classes <- c()
  for (year in start_idx:end_idx){
    classes <- unique(c(classes,tab[,year]))
  }
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  ##only analyse those with DF 2yrs before/after target
  tab.lag <- seqdef(tab, start_idx:end_idx, alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  tabe.seq <- seqecreate(tab.lag, use.labels = FALSE)
  lc <- seqecontain(tabe.seq, event.list = c("DF"))
  tab <- tab[lc,]
  
  print(paste0('Dimensions ori data: ', dim(tab)[1]))
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  tab[,start_idx:(dim(tab)[2]-1)]
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.target.data = tab.nonstable[,start_idx:(dim(tab.nonstable)[2]-1)]
  #names(tab.target.data) = paste0('t',seq(0,dim(tab.target.data)[2]-1))
  names(tab.target.data) = seq(0,dim(tab.target.data)[2]-1)
  
  tab.final = cbind('x'=tab.nonstable$x,'y'=tab.nonstable$y,tab.target.data)
  tab.final$tyear = tyear
    
  print(paste0('Dimensions target data: ', dim(tab.final)[1]))
  
  return(tab.final)
  
}

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
library(labelled)
library(questionr)
library(viridis)
library(tidyr)
library(TraMineR, quietly = TRUE)
library(JLutils, quietly = TRUE)

##settings
tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
geodata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/geodata/postyear/output/clustersv2")
dir.create(geodata_dir, showWarnings = FALSE, recursive = T)
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


##implement cluster##
targetyears = c(2004:2016)

df_all = list()
for (j in targetyears){
  tyear <- j
  df_all[[tyear]] <-subset_target(tyear, lc_target_years)
}

mergeall <- rbindlist(df_all, fill=TRUE)

care_trajectories <- melt(mergeall, id = c("x","y","tyear"), value.name = "LC")
names(care_trajectories) = c("x","y","tyear","lyear","LC")
care_trajectories$lyear <- as.numeric(as.character(care_trajectories$lyear))

#descriptive
describe(care_trajectories, freq.n.max = 10)

care_trajectories <- care_trajectories[complete.cases(care_trajectories$LC), ] 
classes <- unique(care_trajectories$LC)
target_short <- global_shortclasses[sort(classes)]

##plot change by year
ggplot(z) +
  aes(x = lyear) +
  geom_bar()

#status
n <- care_trajectories[lyear %in% (0:17), .(n = .N), by = lyear]$n
etiquettes <- paste0("Y", 0:17, "\n(n=", n, ")")
val_labels(care_trajectories$LC) <- c(
  "DF" = 4,
  "OF" = 5,
  "Fm" = 6,
  "W" = 2,
  "Bu" = 3
)
ggplot(care_trajectories) +
  aes(x = lyear, fill = to_factor(LC)) +
  geom_bar(color = "gray50", width = 1) +
  scale_x_continuous(breaks = 0:17, labels = etiquettes) +
  ggtitle("Distribution of LC types per post-loss year") +
  xlab("") + ylab("") +
  theme_light() +
  theme(legend.position = "bottom") +
  labs(fill = "Land cover") + 
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  guides(fill = guide_legend(nrow = 2))

###Évolution de la cascade de soins au cours du temps
ggplot(care_trajectories) +
  aes(x = lyear, fill = to_factor(LC)) +
  geom_bar(color = "gray50", width = 1, position = "fill") +
  scale_x_continuous(breaks = 0:17, labels = etiquettes) +
  scale_y_continuous(labels = scales::percent) +
  ggtitle("Cascade des soins observée, selon le temps depuis le diagnostic") +
  xlab("") + ylab("") +
  theme_light() +
  theme(legend.position = "bottom") +
  labs(fill = "Statut dans les soins") + 
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  guides(fill = guide_legend(nrow = 2))

#SA
target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

tyear<-"2004-2016"

seq_obj <- create_sequence_object(mergeall, tyear, alphabet, target_short, target_colors)
seq_weights <- attr(seq_obj,"weights")

seqdplot(seq_obj, legend.prop = .25)

sub_cost_method="TRATE"
seq_dist_method="OM"
cluster_method="WARD"

seq_subcost <- seqcost(seq_obj, method = "TRATE",with.missing = FALSE)
seq_dist <- seqdist(seq_obj, method = "OM",indel = seq_subcost$indel, sm = seq_subcost$sm,with.missing = F)

seq_dist_name <- paste0(tyear,'_',sub_cost_method,'_',seq_dist_method)
seq_dist_path <- paste0(seqdist_dir,'/',seq_dist_name)

save_sequence_distances(seq_dist,seq_dist_path)

averageClust <- hclust(as.dist(seq_dist), method = "ward.D", members=seq_weights)

## save target seq
file_path <- paste0(clusters_dir,'/',cluster_method,"_dendogram_",seq_dist_name)
if(!file.exists(paste0(file_path,".RSav"))){
  save(averageClust, file=paste0(file_path,".RSav"))
}

graphics.off()
png(file = paste0(clusters_dir,"/heatmap_",cluster_method,"_",seq_dist_name,".png"), width = 700, height = 700, res = 200)
seq_heatmap(seq_obj, averageClust, labCol=1:18)
dev.off()


##load unique tb
sub_cost_method="TRATE"
seq_dist_method="OM"
cluster_method="WARD"

fn <- '2004-2016_unique_tb.RSav'
load(paste0(seqdata_dir,'/',fn))

fn <- '2004-2016_unique_seq.RSav'
load(paste0(seqdata_dir,'/',fn))
seq_weights <- attr(seq_object,"weights")

maxseq = 14
tyear = paste0('subsetseq',maxseq)

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

n_cluster=4
data.hc <- clusterboot(seq_dist_subset, B=10, distances = TRUE, clustermethod = disthclustCBI, method =
                         "ward.D", k = n_cluster)

data.hc

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

