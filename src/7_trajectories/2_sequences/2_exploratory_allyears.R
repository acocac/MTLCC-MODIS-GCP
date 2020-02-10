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

tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"
proj <- CRS('+proj=longlat +ellps=WGS84')

##Modify next line to your folder
aoi_shp <- readShapeLines(paste0(aux_dir,'/aoi/amazon_raisg.shp'), proj4string=proj)
roads_shp <- readShapeLines(paste0(aux_dir,'/ancillary/roads/roads_2012_gROADSv1.shp'), proj4string=proj)

##analysis with simple classes
targetyears = c(2007:2007)
lc_target_years <-c(2001,2019)

year = 2005

infile <- paste0(indir,'/','resultsseq_tyear_',as.character(year),'_simple.rdata')
results_df = rlist::list.load(infile)

tab.stable = results_df[1][[1]]
tab.nonstable = results_df[2][[1]]

tab.target.seq = tab.nonstable

tab.target.metrics = tab.target.seq

### Longitudinal turbulence and entropy indices
# Computed for each pixel over the time
tab.target.metrics$Entrop <- seqient(tab.target.seq, norm=TRUE, base=exp(1))

png(filename=paste0(chart_dir,"/tentropy_",year,".png"), width = 300, height = 250, units='mm', res=70)
seqHtplot(tab.target.metrics, main = "Entropy", ylab="Entropy index value",xlab=("Time"), legend.prop=0.2)
dev.off()

png(filename=paste0(chart_dir,"/permanence_",year,".png"), width = 300, height = 250, units='mm', res=70)
seqmtplot(tab.target.metrics, with.legend = T, main = "Permanence", ylab="Number of years", legend.prop=0.15)
dev.off()

# #turbulence
# tab.target.seq$Turb <- seqST(tab.target.seq, norm=FALSE)
# 
# # Generate rasters which represent these indices
# xyt <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Turb))
# names(xyt) <- c("x","y","t")
# coordinates(xyt) <- ~ x + y
# gridded(xyt) <- TRUE
# raster_turb <- raster(xyt)
# 
# ##quantile working###
# gyr=colorRampPalette(c("darkgreen","yellow","red"))
# 
# my.at=unique(quantile(tab.target.seq$Turb,seq(0,1,len=5),na.rm=T))
# my.at=c(0,my.at)
# 
# myColorkey <- list(at=my.at, ## where the colors change
#                    labels=as.character(round(my.at,2), ## labels
#                                        at=my.at
#                                        ## where to print labels
#                    ), space="bottom")
# 
# png(filename=paste0(chart_dir,"/turbulenceQ_",year,".png"), width = 600, height = 400, units='mm', res=50)
# # p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
# #p <- levelplot(raster_entrop, layers=1, margin=F, maxpixels = 10e5, par.settings = YlOrRdTheme, at=seq(0, 0.6, length=15), colorkey=list(space="bottom")) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p <- levelplot(raster_turb, layers=1, margin=F, maxpixels = 10e5, col.regions=gyr(length(my.at)), at=my.at, colorkey=myColorkey) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# # p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
# p + layer(sp.lines(roads_shp, lwd=1, col='darkgray', alpha=0.2))
# dev.off()
# 
# ##intervals paper###
# my.at <- c(0,2,3.34,4.35,5.26,6.11,max(tab.target.seq$Turb))
# length(my.at)
# myColorkey <- list(at=my.at, ## where the colors change
#                    labels=as.character(c("","under 2", "2-3.34", "3.34-4.35", "4.35-5.26", "5.26-6.11", "over 6.11%"), ## labels
#                                        at=my.at
#                                        ## where to print labels
#                    ), space="bottom")
# 
# png(filename=paste0(chart_dir,"/turbulenceP_",year,".png"), width = 500, height = 300, units='mm', res=70)
# p <- levelplot(raster_turb, layers=1, margin=F, maxpixels = 10e5, col.regions=gyr(length(my.at)), at=my.at, colorkey=myColorkey) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(roads_shp, lwd=2, col='darkgray', alpha=0.4))
# dev.off()
# 
# ###entropy###
# tab.target.seq$Entrop <- seqient(tab.target.seq, norm=TRUE, base=exp(1))
# 
# # Generate rasters which represent these indices
# xye <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Entrop))
# names(xye) <- c("x","y","e")
# coordinates(xye) <- ~ x + y
# gridded(xye) <- TRUE
# raster_entrop <- raster(xye)
# 
# ##quantile working###
# gyr=colorRampPalette(c("darkgreen","yellow","red"))
# 
# my.at=unique(quantile(tab.target.seq$Entrop,seq(0,1,len=5),na.rm=T))
# my.at=c(0,my.at,1)
#   
# myColorkey <- list(at=my.at, ## where the colors change
#                    labels=as.character(round(my.at,2), ## labels
#                                        at=my.at
#                                        ## where to print labels
#                    ), space="bottom")
# 
# png(filename=paste0(chart_dir,"/entropyQ_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_entrop, layers=1, margin=F, maxpixels = 10e5, col.regions=gyr(length(my.at)), at=my.at, colorkey=myColorkey) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(roads_shp, lwd=1, col='darkgray', alpha=0.2))
# dev.off()
# 
# ##intervals paper###
# my.at <- c(0.17,0.26,0.33,0.38,0.51,1)
# 
# myColorkey <- list(at=my.at, ## where the colors change
#                    labels=as.character(c("under 0.17", "0.17 - 0.26", "0.26-0.33", "0.33-0.38", "0.38-0.51", "over 0.51%"), ## labels
#                      at=my.at
#                      ## where to print labels
#                    ), space="bottom")
# 
# png(filename=paste0(chart_dir,"/entropyP_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_entrop, layers=1, margin=F, maxpixels = 10e5, col.regions=gyr(length(my.at)), at=my.at, colorkey=myColorkey) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(roads_shp, lwd=2, col='darkgray', alpha=0.4))
# dev.off()
# 
# # Calculate the correlation between both indices
# cor(tab.target.seq$Turb,tab.target.seq$Entrop) # 0.9394874
# 
# ## Computes the transition rates
# tr_rates <- seqtrate(tab.target.seq)
# print(tr_rates)
# 
# ###### clustering
# #sample
# sample.tab <- tab.target.seq %>% sample_frac(.2)
# target.final <-tab.target.seq
#   
# ## LCS
# dist.lcs <- seqdist(target.final, method = "LCS", full.matrix=FALSE)
# 
# ## LCP
# dist.lcp <- seqdist(tab.target.seq, method = "LCP") 
# 
# ## Cluster based on LCS
# clusterward_lcs <- hclust(as.dist(dist.lcs),method="ward.D")
# plot(clusterward_lcs)
# cl_lcs <- cutree(clusterward_lcs, k = 5)
# tab.target$clusterlcs <- cl_lcs
# 
# ## Cluster based on LCP
# clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
# plot(clusterward_lcp)
# cl_lcp <- cutree(clusterward_lcp, k = 5)
# tab.target$clusterlcp <- cl_lcp
# 
# # LCS
# seqIplot(tab.target.seq, group = tab.target$clusterlcs, sortv = "from.start")
# # LCP
# seqIplot(tab.target.seq, group = tab.target$clusterlcp, sortv = "from.start")
# 
# ######### Plot clusters' spatial distribution
# # Elaborate raster LCS
# xyz <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target$clusterlcs))
# names(xyz) <- c("x","y","z")
# xyz <- xyz[complete.cases(xyz), ]
# coordinates(xyz) <- ~ x + y
# gridded(xyz) <- TRUE
# raster_clcs <- raster(xyz)
# plot(raster_clcs)
# 
# levels(raster_clcs$z) <- factor(raster_clcs$z)
# 
# png(filename=paste0(chart_dir,"/lcs_",year,".png"), width = 600, height = 400, units='mm', res=50)
# # p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
# p <- levelplot(raster_clcs, layers=1, margin=F, maxpixels = 10e5, att='z', colorkey=list(space="bottom")) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# # p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
# p + layer(sp.lines(roads_shp, lwd=2, col='darkgreen', alpha=0.2))
# dev.off()
# 
# # Elaborate raster LCP
# xyz <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target$clusterlcp))
# names(xyz) <- c("x","y","z")
# xyz <- xyz[complete.cases(xyz), ]
# coordinates(xyz) <- ~ x + y
# gridded(xyz) <- TRUE
# raster_clcp <- raster(xyz)
# plot(raster_clcp)
# 

