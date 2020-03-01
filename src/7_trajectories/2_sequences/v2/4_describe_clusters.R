rm(list = ls())

###start process
library(data.table)
library(TraMineR, quietly = TRUE)
library(ggplot2)
library(labelled)

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
minperiod=12
nclusters=6
  
tyear = paste0(min(targetyears),'-',max(targetyears))  

file_name <- paste0(min(targetyears),'-',max(targetyears),'_minperiod',minperiod,'_unique_seq')
load(paste0(seqdata_dir,'/',file_name,'.RSav'))

cluster_method="WARD" #WCPAMOnce WARD 
sub_cost_method = "TRATE"
seq_dist_method = "OM"

file_name <- paste0(min(targetyears),'-',max(targetyears),'_minperiod',minperiod,'_',sub_cost_method,'_',seq_dist_method)
load(paste0(seqdist_dir,'/',file_name,'.RSav'))

fn <- paste0(cluster_method,'_clustersk',nclusters,'_',min(targetyears),'-',max(targetyears),'_minperiod',minperiod,'_',sub_cost_method,'_',seq_dist_method,'_unique')
load(paste0(clusters_dir,'/',fn,'.RSav'))

# Plot all the sequences within each cluster 
graphics.off()
png(file = paste0(charts_dir,"/desc_seqs_",fn,".png"), width = 1800, height = 1400, res = 200)
seqIplot(seq_obj, group = unique_seq_subset$clusters, ylab=NA, xtlab=1:12, yaxis = F, with.legend = F, border = NA, legend.prop=0.2, cex.axis=0.8, cex.lab=1.3, rows=2,cols=3)
dev.off()

# Plot all the sequences within each cluster 
graphics.off()
png(file = paste0(charts_dir,"/desc_modal_",fn,".png"), width = 2000, height = 1200, res = 200)
seqmsplot(seq_obj, group = unique_seq_subset$clusters, ylab=NA, xtlab=1:12, yaxis = T, with.legend = F, border = NA, legend.prop=0.2, cex.axis=0.9, cex.lab=1.3, rows=2,cols=3)
dev.off()

graphics.off()
png(file = paste0(charts_dir,"/desc_permanence_",fn,".png"), width = 500, height = 800, res = 100)
seqmtplot(seq_obj, group = unique_seq_subset$clusters,  border = NA)
dev.off()

graphics.off()
png(file = paste0(charts_dir,"/desc_representatives_",fn,".png"), width = 500, height = 800, res = 100)
seqrplot(seq_obj, group = unique_seq_subset$clusters,  border = NA, dist.matrix = seq_dist, criterion = "dist")
dev.off()

levels(unique_seq_subset$clusters) <- c("one","two","three")


graphics.off()
png(file = paste0(charts_dir,"/desc_entropy_",fn,".png"), width = 1900, height = 1200, res = 200)
seqHtplot(seq_obj, group = unique_seq_subset$clusters,   ylab="Shannon entropy index [0-1]", xtlab=1:12, yaxis = T, with.legend = F, border = NA, legend.prop=0.2, cex.axis=0.9, cex.lab=1.3, rows=2,cols=3)
dev.off()


#entropy
stat.bf <- seqstatd(seq_obj)
ent <- stat.bf$Frequencies


seqHtplot(seq, group = seq.part, xtlab = 14:50)

unique_seq$groupe <- factor(
  large_m18$typo_pam,
  c(85, 23, 410, 6),
  c("Rapides", "Lents", "Inaboutis", "Hors soins")
)

###all
unique_seq_subset$ordre_cmd <- cmdscale(as.dist(seq_dist), k = 1)

# name groups
unique_seq_subset$groups <- factor(
  unique_seq_subset$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("DF>W", "DF>OF", "DF>OF>DF", "DF>Fm>OF","DF>Fm","DF>OF>Fm")
)

# calculer le rang des individus dans chaque groupe
setorder(unique_seq_subset, "ordre_cmd")
unique_seq_subset[, rang_cmd := 1:.N, by = groups]

target_reshape <- melt(unique_seq_subset, id = c("groups","rang_cmd"), measure.vars=3:13,value.name = "LC")

classes <- unique(target_reshape$LC)

target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]
target_long <- global_classes[sort(classes)]

# créer un fichier long
long_m18 <- care_trajectories[id %in% large_m18$id & month <= 18]
long_m18 <- merge(
  long_m18,
  large_m18[, .(id, groupe, rang_cmd)],
  by = "id",
  all.x = TRUE
)
long_m18$care_statusF <- to_factor(long_m18$care_status)

# calculer les effectifs par groupe
tmp <- unique_seq_subset[, .(n = .N), by = groups]
tmp[, groupe_n := paste0(groupe, "\n(n=", n, ")")]
long_m18 <- merge(
  long_m18,
  tmp[, .(groupe, groupe_n)],
  by = "groupe",
  all.x = TRUE
)

summary(target_reshape)

target_reshape$LCf <- to_factor(target_reshape$LC)

library(viridis)

# graphique des tapis de séquences
ggplot(target_reshape) +
  aes(x = variable, y = rang_cmd, fill = LCf) +
  geom_raster() +
  facet_grid(groups ~ ., space = "free", scales = "free") +
  #scale_x_continuous(breaks = 0:11, labels = paste0("Y", 1:12)) +
  scale_y_continuous(minor_breaks = NULL) +
  xlab("") + ylab("") +
  theme_light() +
  theme(legend.position = "bottom") +
  labs(fill = "Statut dans les soins") + 
  scale_fill_manual("",values=target_colors,breaks=sort(classes),
                    labels=target_long) +
  guides(fill = guide_legend(nrow = 2))
