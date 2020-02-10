
##TODO 
#find optimal clusters

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


tile <- 'tile_raisg'
year <- 2004

#dirs
indir <- paste0("E:/acocac/research/",tile,"/post/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/post/charts")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)

#files
infile <- paste0(indir,'/','resultsraw_tyear_',as.character(year),'.rdata')

#settings
lc_target_years <-c(2001,2017)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
proj <- CRS('+proj=longlat +ellps=WGS84')

##Modify next line to your folder
aoi_shp <- readShapeLines(paste0(aux_dir,'/aoi/amazon_raisg.shp'), proj4string=proj)
roads_shp <- readShapeLines(paste0(aux_dir,'/ancillary/roads/roads_2012_gROADSv1.shp'), proj4string=proj)

##analysis

results_df = rlist::list.load(infile)

global_classes = results_df[1][[1]]
global_shortclasses =  results_df[2][[1]]
global_colors =  results_df[3][[1]]
classes = results_df[4][[1]]
tab = results_df[5][[1]]
tyear = results_df[6][[1]]

target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

labels <- target_classes                             
short_labels <- target_short
palette <- target_colors

#yearls <- paste0("c",as.character(seq(lc_target_years[1],lc_target_years[2],1)))
yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
pos_year <- which(yearls == tyear)

#create stratas
std.df <- apply(tab[,(2+pos_year-2):(dim(tab)[2]-1)], 1, sd) 

tab.stable = tab[std.df==0,]
tab.nonstable = tab[std.df!=0,]

tab.target = tab.nonstable

tab.target.seq <- seqdef(tab.target, (2+pos_year-2):(length(yearls)+2), alphabet = alphabet, states = short_labels,
                            cpal = palette, labels = short_labels)

aggtarget <- wcAggregateCases(tab.target[, 3:(length(yearls)+2)])
uniquetarget <- tab.target[aggtarget$aggIndex, ]

#target_seq_unique <- seqdef(uniquetarget, 3:(length(yearls)+2), weights = aggtarget$aggWeights, alphabet = alphabet, states = short_labels, cpal = target_colors.s,with.missing = TRUE)
target_seq_unique <- seqdef(uniquetarget, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels, cpal = target_colors.s,with.missing = TRUE)

target.final = target_seq_unique

# diss_target <- seqdist(target_seq_unique, method = "DHD")
# 
# ## Aglomerative hierarchical clustering
# ## First of all we employ different methods to select the optimal number of clusters
# graphics.off() 
# averageClust1 <- hclust(as.dist(diss_target), method = "average", members = aggtarget$aggWeights)
# 
# avgClustQual1 <- as.clustrange(averageClust1, diss_target, weights = aggtarget$aggWeights, ncluster = 10)

## LCS
dist.lcs <- seqdist(target.final, method = "LCS")

## LCP
dist.lcp <- seqdist(target.final, method = "LCP") 

## Cluster based on LCS
clusterward_lcs <- hclust(as.dist(dist.lcs),method="ward.D")
#plot(clusterward_lcs)
cl_lcs <- cutree(clusterward_lcs, k = 5)
uniquetarget$clusterlcs <- cl_lcs

## Cluster based on LCP
clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
cl_lcp <- cutree(clusterward_lcp, k = 5)
uniquetarget$clusterlcp <- cl_lcp

# LCS
seqIplot(target.final, group = uniquetarget$clusterlcs, sortv = "from.start")
# LCP
seqIplot(target.final, group = uniquetarget$clusterlcp, sortv = "from.start")

##merge tables
clusterlcs_df <- tab.target %>% 
  left_join(select(uniquetarget, clusterlcs, sec), by = "sec")


######### Plot clusters' spatial distribution
# Elaborate raster LCS
xyz <- as.data.frame(cbind(clusterlcs_df$x,clusterlcs_df$y,clusterlcs_df$clusterlcs))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcs <- raster(xyz)
plot(raster_clcs)

myraster<-ratify(raster_clcs)
rat <- levels(myraster)[[1]]
rat$classes <- c(1:5)
levels(myraster)<-rat

my_palette <- brewer.pal(n = 5, name = "Set1")

png(filename=paste0(chart_dir,"/lcs_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
p <- levelplot(myraster, layers=1, margin=F, maxpixels = 10e5, att='classes', colorkey=list(space="bottom"), col.regions = my_palette) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
p + layer(sp.lines(roads_shp, lwd=0.5, col='darkgray', alpha=0.2))
dev.off()

tabe.seq <- seqecreate(tab.target.seq, use.labels = FALSE)
forest <- seqecontain(tabe.seq, event.list = c("F"))
forest_tab <- tab.target[forest,]

mosaic.seq <-  seqdef(forest_tab, (2+pos_year):(length(yearls)+2), alphabet = alphabet, states = short_labels,
                      cpal = palette, labels = short_labels)

mosaic.seq
#create stratas
std.df <- apply(forest_tab[,(2+pos_year):(length(yearls)+2)], 1, sd) 

tab.stable1 = forest_tab[std.df==0,]
tab.nonstable1 = forest_tab[std.df!=0,]

mosaic.seqe <- seqecreate(mosaic.seq, use.labels = FALSE)

fsubseq <- seqefsub(mosaic.seqe, pmin.support = 0.0510)
plot(fsubseq[1:10], col = "grey98")

