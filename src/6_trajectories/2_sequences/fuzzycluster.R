rm(list = ls())

library(TraMineR)
library(reshape2)
library(tidyr)
library(plyr)
library(rlist)
library(dplyr)
library(WeightedCluster)

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
fuzzy_dir <- paste0("E:/acocac/research/",tile,"/trajectories/fuzzy")
dir.create(fuzzy_dir, showWarnings = FALSE, recursive = T)

fn <- '2012_TRATE_OM.RSav'
load(paste0(seqdist_dir,'/',fn))

fn <- '2012_unique_seq.RSav'
load(paste0(seqdata_dir,'/',fn))


fclust <- fanny(seq_dist, k=7, diss=TRUE, memb.exp = 1.5)

graphics.off()
png(file = paste0(fuzzy_dir,"/test.png"), width = 500, height = 1200, res = 100)
fuzzyseqplot(seq_object, group=fclust, type="d")
dev.off()

