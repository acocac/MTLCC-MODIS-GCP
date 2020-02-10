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


tile <- 'tile_raisg'
year <- 2006

#dirs
indir <- paste0("E:/acocac/research/",tile,"/post/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/post/pngs")
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
pos_year <- (which(yearls == tyear)-1)

#order distriplot by major
tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                  cpal = palette, labels = short_labels, with.missing = TRUE)
## Get state freq with seqmeant
mt <- seqmeant(tab.seq)
## order of frequencies
ord <- order(mt, decreasing = TRUE)

## Sorted alphabet
alph.s <- rownames(mt)[ord]
## we need also to sort accordingly labels and colors
target_classes.s <- target_classes[ord]
target_short.s <- target_short[ord]
target_colors.s <- target_colors[ord]

## Define sequence object with sorted states
tab.seq.s <- seqdef(tab.seq, alphabet = alph.s, states = target_short.s,
                    labels = target_classes.s, cpal = target_colors.s,with.missing = TRUE)

#create stratas
std.df <- apply(tab[,3:(dim(tab)[2]-1)], 1, sd) 

tab.stable = tab[std.df==0,]
tab.nonstable = tab[std.df!=0,]

tab.target = tab.nonstable

tab.target.seq <- seqdef(tab.target, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                            cpal = palette, labels = short_labels)

### Longitudinal turbulence and entropy indices 
# Computed for each pixel over the time
tab.target.seq$Turb <- seqST(tab.target.seq, norm=FALSE)
tab.target.seq$Entrop <- seqient(tab.target.seq, norm=TRUE, base=exp(1))

seqmtplot(tab.target.seq, with.legend = T, main = "Permanence", ylab="Number of years", legend.prop=0.2)
seqHtplot(tab.target.seq, main = "Entropy", ylab="Entropy index value",xlab=("Time"), legend.prop=0.2)

# Generate rasters which represent these indices
xyt <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Turb))
names(xyt) <- c("x","y","t")
coordinates(xyt) <- ~ x + y
gridded(xyt) <- TRUE
raster_turb <- raster(xyt)
#plot(raster_turb)
#plot including median per-axis
# levelplot(raster_turb, layers = 1, margin = list(FUN = 'median', axis=TRUE), contour=TRUE)

png(filename=paste0(chart_dir,"/turbulence_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
p <- levelplot(raster_turb, layers=1, margin=F, maxpixels = 10e5, colorkey=list(space="bottom"))
# p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
p + layer(sp.lines(roads_shp, lwd=2, col='darkgreen', alpha=0.2))
dev.off()

# Generate rasters which represent these indices
xye <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Entrop))
names(xye) <- c("x","y","e")
coordinates(xye) <- ~ x + y
gridded(xye) <- TRUE
raster_entrop <- raster(xye)
plot(raster_entrop)

png(filename=paste0(chart_dir,"/entropy_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
p <- levelplot(raster_entrop, layers=1, margin=F, maxpixels = 10e5, par.settings = RdBuTheme, at=seq(0, 1, length=15), colorkey=list(space="bottom")) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
p + layer(sp.lines(roads_shp, lwd=2, col='darkgreen', alpha=0.2))
dev.off()


# Calculate the correlation between both indices
cor(tab.target.seq$Turb,tab.target.seq$Entrop) # 0.9394874

## Computes the transition rates
tr_rates <- seqtrate(tab.target.seq)
print(tr_rates)

###### clustering
## LCS
dist.lcs <- seqdist(tab.target.seq, method = "LCS")

## LCP
dist.lcp <- seqdist(tab.target.seq, method = "LCP") 

## Cluster based on LCS
clusterward_lcs <- hclust(as.dist(dist.lcs),method="ward.D")
plot(clusterward_lcs)
cl_lcs <- cutree(clusterward_lcs, k = 5)
tab.target$clusterlcs <- cl_lcs

## Cluster based on LCP
clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
plot(clusterward_lcp)
cl_lcp <- cutree(clusterward_lcp, k = 5)
tab.target$clusterlcp <- cl_lcp

# LCS
seqIplot(tab.target.seq, group = tab.target$clusterlcs, sortv = "from.start")
# LCP
seqIplot(tab.target.seq, group = tab.target$clusterlcp, sortv = "from.start")

######### Plot clusters' spatial distribution
# Elaborate raster LCS
xyz <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target$clusterlcs))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcs <- raster(xyz)
plot(raster_clcs)

levels(raster_clcs$z) <- factor(raster_clcs$z)

png(filename=paste0(chart_dir,"/lcs_",year,".png"), width = 600, height = 400, units='mm', res=50)
# p <- levelplot(raster_turb, layers=1, margin = list(FUN = median, axis=TRUE), maxpixels = 10e5)
p <- levelplot(raster_clcs, layers=1, margin=F, maxpixels = 10e5, att='z', colorkey=list(space="bottom")) #https://stackoverflow.com/questions/16847377/how-to-set-the-maximum-and-minimum-values-for-a-lattice-plot
# p + layer(sp.lines(aoi_shp, lwd=0.8, col='darkgray'))
p + layer(sp.lines(roads_shp, lwd=2, col='darkgreen', alpha=0.2))
dev.off()

# Elaborate raster LCP
xyz <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target$clusterlcp))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcp <- raster(xyz)
plot(raster_clcp)


