## Multi-channel sequence Analysis

rm(list = ls())

## Load required libraries
library(TraMineR)
library(foreign)
library(WeightedCluster)
library(dplyr)
library(RColorBrewer)
#library(factoextra)
library(NbClust)
library(ggplot2)

tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts_allyears")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)

##analysis with simple classes
targetyears = c(2004:2006)
lc_target_years <-c(2001,2019)

tyear <- 2004

infile <- paste0(indir,'/','resultsraw_tyear_',as.character(tyear),'_raw.rdata')

results_df = rlist::list.load(infile)

global_classes = results_df[1][[1]]
global_shortclasses =  results_df[2][[1]]
global_colors =  results_df[3][[1]]
classes = results_df[4][[1]]
tab = results_df[5][[1]]

target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
pos_year <- which(yearls == tyear)

#create stratas
std.df <- apply(tab[,2:(dim(tab)[2]-1)], 1, sd) 

tab.stable = tab[std.df==0,]
tab.nonstable = tab[std.df!=0,]

tab.target = tab.nonstable

bf <- as.matrix(tab[, 2:(dim(tab)[2]-1)])
children <- bf == 4
married <- bf == 5

diss_life <- seqdef(children, cnames = 15:30, cpal = brewer.pal(6,
                                                                "Paired")[1:2])
diss_work <- seqdef(married, cnames = 15:30, cpal = brewer.pal(6,
                                                              "Paired")[3:4])     

par(mfrow = c(3, 1))
seqIplot(child.seq, with.legend = FALSE, title = "With Children",
         sortv = "from.start")
seqIplot(marr.seq, with.legend = FALSE, title = "Married",
         sortv = "from.start")

mcdiss <- seqdistmc(channels = list(diss_life, diss_work), method = "DHD")

## Aglomerative hierarchical clustering
## First of all we employ different methods to select the optimal number of clusters
graphics.off()
averageClust <- hclust(as.dist(mcdiss), method = "average")
averageClust1 <- hclust(as.dist(diss_life), method = "average", members = agglife$aggWeights)
averageClust2 <- hclust(as.dist(diss_work), method = "average", members = aggwork$aggWeights)
avgClustQual <- as.clustrange(averageClust, mcdiss, ncluster = 10)
avgClustQual1 <- as.clustrange(averageClust1, diss_life, weights = agglife$aggWeights, ncluster = 10)
avgClustQual2 <- as.clustrange(averageClust2, diss_work, weights = aggwork$aggWeights, ncluster = 10)
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/F-w_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual, norm = "zscore", withlegend = F)
dev.off()
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Family_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual1, norm = "zscore", withlegend = F)
dev.off()
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Work_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual2, norm = "zscore", withlegend = F)
dev.off()
summary(avgClustQual, max.rank = 2)
summary(avgClustQual1, max.rank = 2)
summary(avgClustQual2, max.rank = 2)

## Create sequence objects for both dimensions
life_seq <- seqlist[1][[1]]
work_seq <- seqlist[2][[1]]
life <- tablist[1][[1]]
work <- tablist[2][[1]]
agglife <- wcAggregateCases(life[, 3:ncol(life)-1])
aggwork <- wcAggregateCases(work[, 3:ncol(work)-1])
uniquelife <- life[agglife$aggIndex, 3:ncol(life)]
uniquework <- work[aggwork$aggIndex, 3:ncol(life)]

life_seq_unique <- seqdef(uniquelife, 1:(ncol(uniquelife)-1), weights = agglife$aggWeights, alphabet = alphabetlist[1][[1]], states = short_labelslist[1][[1]],
                          cpal = palettelist[1][[1]], labels = short_labelslist[1][[1]])
work_seq_unique <- seqdef(uniquework, 1:(ncol(uniquework)-1), weights = aggwork$aggWeights, alphabet = alphabetlist[2][[1]], states = short_labelslist[2][[1]],
                          cpal = palettelist[2][[1]], labels = short_labelslist[2][[1]])

diss_life <- seqdist(life_seq_unique, method = "DHD")
diss_work <- seqdist(work_seq_unique, method = "DHD")

## Multichannel distances
mcdiss <- seqdistmc(list(life_seq, work_seq), method = "DHD", full.matrix = TRUE, with.missing=TRUE)

## Aglomerative hierarchical clustering
## First of all we employ different methods to select the optimal number of clusters
graphics.off()
averageClust <- hclust(as.dist(mcdiss), method = "average")
averageClust1 <- hclust(as.dist(diss_life), method = "average", members = agglife$aggWeights)
averageClust2 <- hclust(as.dist(diss_work), method = "average", members = aggwork$aggWeights)
avgClustQual <- as.clustrange(averageClust, mcdiss, ncluster = 10)
avgClustQual1 <- as.clustrange(averageClust1, diss_life, weights = agglife$aggWeights, ncluster = 10)
avgClustQual2 <- as.clustrange(averageClust2, diss_work, weights = aggwork$aggWeights, ncluster = 10)
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/F-w_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual, norm = "zscore", withlegend = F)
dev.off()
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Family_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual1, norm = "zscore", withlegend = F)
dev.off()
png(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Work_opt_clusters.png", width = 560, height = 474, res = 99)
plot(avgClustQual2, norm = "zscore", withlegend = F)
dev.off()
summary(avgClustQual, max.rank = 2)
summary(avgClustQual1, max.rank = 2)
summary(avgClustQual2, max.rank = 2)

## In view of the graph it seems that 8 is our optimal number of clusters
## Alternatively, we conduct PAM clustering
pamClustRange <- wcKMedRange(mcdiss, kvals = 2:10)

## Plot the clusters
par(mar=c(1,1,1,1))
clusterward <- agnes(mcdiss, diss = T, method = "ward")
family_work <- cutree(clusterward, k = 8)
labs <- factor(family_work, labels = paste("Cluster", 1:8))
pdf(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Family_clusters.pdf", 
    width = 25, height = 40)
seqdplot(life_seq, group = labs, border = NA, withlegend = F)
dev.off()
pdf(file = "E:/acocac/research/scripts/sequences_repos/TFG-ADE-2017/temp/Work_clusters.pdf", 
    width = 25, height = 40)
seqdplot(work_seq, group = labs, border = NA, withlegend = F)
dev.off()

rm(diss_life, diss_work, life_seq, life_seq_unique, mcdiss, uniquelife,
   uniquework, work_seq, work_seq_unique, agglife, aggwork, averageClust, averageClust1,
   averageClust2, avgClustQual, avgClustQual1, avgClustQual2, clusterward, pamClustRange)

