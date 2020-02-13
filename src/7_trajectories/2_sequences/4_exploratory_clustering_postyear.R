##TODO
# * best n clusters
# * 

rm(list = ls())

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


# targetyears = c(2004:2018)

tile <- 'AMZ'
tyear <- 2004

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts_postyear")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)
geodata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/geodata/postyear/output")
dir.create(geodata_dir, showWarnings = FALSE, recursive = T)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
proj <- CRS('+proj=longlat +ellps=WGS84')

##Modify next line to your folder
aoi_shp <- readShapeLines(paste0(aux_dir,'/aoi/amazon_raisg.shp'), proj4string=proj)
roads_shp <- readShapeLines(paste0(aux_dir,'/ancillary/roads/roads_2012_gROADSv1.shp'), proj4string=proj)
dem <- raster(paste0(aux_dir,'/ancillary/gee/srtm.tif'))
slope <- raster(paste0(aux_dir,'/ancillary/gee/slope.tif'))
access <- raster(paste0(aux_dir,'/ancillary/gee/access.tif'))

##analysis with simple classes
lc_target_years <-c(2001,2019)

##analysis
infile <- paste0(indir,'/','resultsseq_tyear_',as.character(tyear),'_simple.rdata')

results_df = rlist::list.load(infile)

tab = results_df[4][[1]]
tyear = results_df[3][[1]]

yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))

### Determine the number of different sequences
unique_traj <- sort(unique(tab$sec))
length(unique_traj)  # 

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

##add and prepare aux columns
tab.target_geo = tab.target
coordinates(tab.target_geo) <- c("x", "y")
mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))

dem_val =raster::extract(dem, mypoints)
slope_val =raster::extract(slope, mypoints)
access_val =raster::extract(access, mypoints)

tab.target$dem = dem_val
tab.target$slope = slope_val
tab.target$access = access_val

## Create categorical data from covariates
tab.target$demK<-cut(tab.target$dem, c(50,100,500,1200))
tab.target$slopeK<-cut(tab.target$slope, c(0,2,6,25))
## Two categories for distance and slope (binary variable)
tab.target$accessK2<-cut(tab.target$access, c(0,500,1000))
tab.target$slopeK2<-cut(tab.target$slope, c(-1,6,25))

tab.nonstable.seq <- seqdef(tab.nonstable, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                            cpal = target_colors, labels = target_short)

tab.target.seq = tab.nonstable.seq

tab.target.metrics = tab.target.seq

aggtarget <- wcAggregateCases(tab.target[,start_idx:(length(yearls)+2)])
uniquetarget <- tab.target[aggtarget$aggIndex, ]

target_seq_unique <- seqdef(uniquetarget, start_idx:(length(yearls)+2), weights = aggtarget$aggWeights, alphabet = alphabet, states = target_short, cpal = target_colors,with.missing = TRUE)

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

##extras###
# ##averagetree - only work in pc with admin rights
# averageTree <- as.seqtree(averageClust, seqdata=target_seq_unique, diss=dist.om1, ncluster=6)
# ## Graphical representation of the tree (you need to have Graphviz installed before lauchning R)
# seqtreedisplay(averageTree, type="d", border=NA,  showdepth=TRUE) ##only working on laptop
# 
# ## Compute PAM clustering and cluster quality measure for different number of groups (ranging from 2 to 10)
# pamClustRange <- wcKMedRange(dist.om1, kvals=2:10, weights=aggtarget$aggWeights)
# 
# ## Print the 2 best number of group according to each quality measure for the PAM clustering
# summary(pamClustRange, max.rank=2)
# 
# ## The best clustering was found using average clustering in 5 groups according to ASW (average silhouette width)
# seqdplot(target_seq_unique, group=avgClustQual$clustering$cluster5, border=NA)
# 
# ## Clustering was made on distinct sequences
# ## Recover the clustering solution in the original (full) dataset
# uniqueCluster5 <- avgClustQual$clustering$cluster5
# tab.target$cluster5 <- uniqueCluster5[aggtarget$disaggIndex]
# 

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
png(file = paste0(chart_dir,"/clusters_OM1_seq_def",tyear,".png"), width = 600, height = 800, res = 90)
seqIplot(target_seq_unique, group = uniquetarget$clusterom1, sortv = "from.start", with.legend = F)
dev.off()


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
png(file = paste0(chart_dir,"/clusters_OM1_seq_def",tyear,".png"), width = 600, height = 800, res = 90)
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
