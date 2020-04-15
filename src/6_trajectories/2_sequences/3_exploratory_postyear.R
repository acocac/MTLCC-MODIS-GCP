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
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts_postyear")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)
geodata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/geodata/postyear/input")
dir.create(geodata_dir, showWarnings = FALSE, recursive = T)

##analysis with simple classes
targetyears = c(2004:2018)
lc_target_years <-c(2001,2019)

for (i in 1:length(targetyears)){
  
  infile <- paste0(indir,'/','resultsseq_tyear_',as.character(targetyears[i]),'_simple.rdata')
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
  
  tab.seq <- seqdef(tab, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.nonstable.seq <- seqdef(tab.nonstable, start_idx:(length(yearls)+2), alphabet = alphabet, states = target_short,
                              cpal = target_colors, labels = target_short)
  
  tab.target = tab.nonstable
  tab.target.seq=tab.nonstable.seq
  
  tab.target.metrics = tab.target.seq
  
  ### Longitudinal turbulence and entropy indices
  # Computed for each pixel over the time
  png(filename=paste0(chart_dir,"/permanence_",tyear,".png"), width = 150, height = 150, units='mm', res=70)
  # seqmtplot(tab.target.metrics, with.legend = F, main = "Permanence", ylab="Number of years", legend.prop=0.15)
  seqmtplot(tab.target.metrics, with.legend = F, main = as.character(tyear), ylab="Number of years following deforestation")
  dev.off()
  
  #turbulence
  tab.target.seq$Turb <- seqST(tab.target.seq, norm=TRUE)
  
  # Generate rasters which represent these indices
  xyt <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Turb))
  names(xyt) <- c("x","y","t")
  coordinates(xyt) <- ~ x + y
  gridded(xyt) <- TRUE
  raster_turb <- raster(xyt)
  crs(raster_turb) <- CRS('+init=EPSG:4326')
  
  writeRaster(raster_turb,paste0(geodata_dir,"/turbulence_",tyear,".tif"), format="GTiff", overwrite=TRUE)
  
  ###entropy###
  tab.target.seq$Entrop <- seqient(tab.target.seq, norm=TRUE, base=exp(1))

  # Generate rasters which represent these indices
  xye <- as.data.frame(cbind(tab.target$x,tab.target$y,tab.target.seq$Entrop))
  names(xye) <- c("x","y","e")
  coordinates(xye) <- ~ x + y
  gridded(xye) <- TRUE
  raster_entrop <- raster(xye)
  crs(raster_entrop) <- CRS('+init=EPSG:4326')

  writeRaster(raster_entrop,paste0(geodata_dir,"/entropy_",tyear,".tif"), format="GTiff", overwrite=TRUE)
}

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

