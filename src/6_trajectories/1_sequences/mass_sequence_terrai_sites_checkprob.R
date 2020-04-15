##### Analysis of High Temporal Resolution Land Use/Land Cover Trajectories
##### Supplementary material to article published in Land https://www.mdpi.com/journal/land
##### R script to carry out the analysis presented in the paper
##### JF Mas - Universidad Nacional Autónoma de México - jfmas@ciga.unam.mx


# Working directory and libraries
library(TraMineR)
library(raster)
library(fastcluster)
library(tidyr)
library(WeightedCluster)
library(ggplot2)
library(reshape2)

##Function Calculate Statistics##
calc_seq<- function(scheme, lucscheme, year, yearls){
  
  # scheme <- 'raw'
  # target_year <-c(2004,2014)
  # yearls <- paste0("c",as.character(seq(2004,2017,1)))
  
  files <- list.files(path=dir, pattern="*.tif$", all.files=FALSE, full.names=TRUE,recursive=TRUE)
  
  files <- Filter(function(x) grepl(pattern, x), files)
  
  lucserie <- raster::stack(files)
  
  # plot(lucserie[[1]])
  
  terrai_file <- 'T:/GISDATA_terra/outputs/Latin/2004_01_01_to_2019_04_07/TIFF/GEOGRAPHIC/WGS84/decrease/region_lat/classified/latin_decrease_2004_01_01_to_2019_04_07.tif'
  terrai_raster <- raster(terrai_file)
  
  e <- extent(lucserie)
  
  terrai.crop <- crop(terrai_raster, e)
  # plot(terrai.crop)
  
  # target_year <- year
  # target.values <- function(x) { x[x!=target_year] <- NA; return(x) }
  # terrai_target <- calc(terrai.crop, target.values)
  # 
  Syear <- year[1]
  Eyear <- year[2]
  target.values <- function(x) { x[x<Syear] <- NA; return(x) }
  terrai_target <- calc(terrai.crop, target.values)
  target.values <- function(x) { x[x>Eyear] <- NA; return(x) }
  terrai_target <- calc(terrai_target, target.values)
  target.values <- function(x) { x[!is.na(x)] <- 1; return(x) }
  terrai_target <- calc(terrai_target, target.values)
  
  plot(terrai_target, legend=FALSE, col=gray(0:100/100))
  
  serie_raw <- mask(x=lucserie, mask=terrai_target)

  
  if (scheme == 'raw'){
    serie <- serie_raw
    if (lucscheme == 'copernicus') {
      global_classes <- c('Closed forest evergreen needleleaf',
                          'Closed forest deciduous needleleaf',
                          'Closed forest evergreen broadleaf',
                          'Closed forest deciduous broadleaf',
                          'Closed forest mixed',
                          'Closed forest unknown',
                          'Open forest evergreen needleleaf',
                          'Open forest deciduous needleleaf',
                          'Open forest evergreen broadleaf',
                          'Open forest deciduous broadleaf',
                          'Open forest mixed',
                          'Open forest unknown',
                          'Shrubs',
                          'Herbaceous vegetation',
                          'Herbaceous wetland',
                          'Moss and lichen',
                          'Bare - sparse vegetation',
                          'Cultivated and managed vegetation-agriculture cropland',
                          'Urban - built up',
                          'Snow and Ice',
                          'Permanent water bodies',
                          'Open sea')
      
      global_shortclasses <- c('CFEN',
                               'CFDN',
                               'CFEB',
                               'CFDN',
                               'CFX',
                               'CFU',
                               'OFEN',
                               'OFDN',
                               'OFEB',
                               'OFDN',
                               'OFX',
                               'OFU',
                               'S',
                               'HV',
                               'HW',
                               'ML',
                               'Ba',
                               'C',
                               'Bu',
                               'SI',
                               'PW',
                               'OS')
      
      global_colors = c('#58481f', 
                        '#70663e',
                        '#009900',
                        '#00cc00',
                        '#4e751f',
                        '#007800',
                        '#666000',
                        '#8d7400',
                        '#8db400',
                        '#a0dc00',
                        '#929900',
                        '#648c00',
                        '#ffbb22',
                        '#ffff4c',
                        '#0096a0',
                        '#fae6a0',
                        '#b4b4b4',
                        '#f096ff',
                        '#fa0000',
                        '#f0f0f0',
                        '#0032c8',
                        '#000080')
    } 
    else if (lucscheme == 'MCD12Q1v6LCType1') {
      global_classes <- c('Evergreen needleleaf forest', 'Evergreen broadleaf forest',
                          'Deciduous needleleaf forest', 'Deciduous broadleaf forest',
                          'Mixed forest', 'Closed shrublands', 'Open shrublands',
                          'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                          'Croplands', 'Urban and built-up', 'Cropland natural vegetation mosaic',
                          'Snow and ice', 'Barren or sparsely vegetated', 'Water')
      
      global_shortclasses <- c('ENF', 'EBF',
                               'DNF', 'DBF',
                               'MF', 'CS', 'OS',
                               'WS', 'S', 'G', 'PW',
                               'C', 'Bu', 'CN',
                               'SI', 'Ba', 'W')
      
      global_colors = c('#05450a', '#086a10',
                        '#54a708',
                        '#78d203',
                        '#009900',
                        '#c6b044',
                        '#dcd159',
                        '#dade48',
                        '#fbff13',
                        '#b6ff05',
                        '#27ff87',
                        '#c24f44',
                        '#a5a5a5',
                        '#ff6d4c',
                        '#69fff8',
                        '#f9ffa4',
                        '#1c0dff')
    }
    else if (lucscheme == 'MCD12Q1v6LCProp1') {
      global_classes <- c('Barren',
                          'Permanent Snow and Ice',
                          'Water Bodies',
                          'Evergreen Needleleaf Forests',
                          'Evergreen Broadleaf Forests',
                          'Deciduous Needleleaf Forests',
                          'Deciduous Broadleaf Forests',
                          'Mixed Broadleaf-Needleleaf Forests',
                          'Mixed Broadleaf Evergreen-Deciduous Forests',
                          'Open Forests',
                          'Sparse Forests',
                          'Dense Herbaceous',
                          'Shrubs',
                          'Sparse Herbaceous',
                          'Dense Shrublands',
                          'Shrubland-Grassland Mosaics',
                          'Sparse Shrublands')
      
      global_shortclasses <- c('Ba', 'SI',
                               'W', 'ENF',
                               'EBF', 'DNF', 'DBF',
                               'MFN', 'MFED', 'OF',
                               'SF', 'DH', 'S', 'SH',
                               'DS', 'SGM', 'SS')
      
      global_colors = c('#f9ffa4','#69fff8','#1c0dff',
                        '#05450a','#086a10','#54a708',
                        '#78d203','#005a00','#009900',
                        '#52b352','#00d000','#b6ff05',
                        '#98d604','#dcd159','#f1fb58',
                        '#fbee65')
    }   
    else if (lucscheme == 'MCD12Q1v6LCProp2') {
      global_classes <- c('Barren',
                          'Permanent Snow and Ice',
                          'Water Bodies',
                          'Urban and Built-up Lands',
                          'Dense Forests',
                          'Open Forests',
                          'Forest/Cropland Mosaics',
                          'Natural Herbaceous',
                          'Natural Herbaceous-Croplands Mosaics',
                          'Herbaceous Croplands',
                          'Shrublands')
      
      global_shortclasses <- c('Ba', 'SI',
                               'W', 'Bu',
                               'DF', 'OF', 'FCM',
                               'NH', 'NHCM', 'HC',
                               'S')
      
      global_colors = c('#f9ffa4','#69fff8','#1c0dff',
                        '#a5a5a5','#003f00','#006c00',
                        '#e3ff77','#b6ff05','#93ce04',
                        '#77a703','#dcd159')
    }   
    else if (lucscheme == 'esa') {
      global_classes <- c('Cropland rainfed',
                          'Cropland rainfed Herbaceous cover',
                          'Cropland rainfed Tree or shrub cover',
                          'Cropland irrigated or post-flooding',
                          'Mosaic cropland gt 50 natural vegetation (tree/shrub/herbaceous cover) lt 50',
                          'Mosaic natural vegetation gt 50 cropland lt 50',
                          'Tree cover broadleaved evergreen closed to open gt 15',
                          'Tree cover  broadleaved  deciduous  closed to open gt 15',
                          'Tree cover  broadleaved  deciduous  closed gt 40',
                          'Tree cover  broadleaved  deciduous  open 15 to 40',
                          'Tree cover  needleleaved  evergreen  closed to open gt 15',
                          'Tree cover  needleleaved  evergreen  closed gt 40',
                          'Tree cover  needleleaved  evergreen  open 15 to 40',
                          'Tree cover  needleleaved  deciduous  closed to open gt 15',
                          'Tree cover  needleleaved  deciduous  closed gt 40',
                          'Tree cover  needleleaved  deciduous  open 15 to 40',
                          'Tree cover  mixed leaf type',
                          'Mosaic tree and shrub gt 50 herbaceous cover lt 50',
                          'Mosaic herbaceous cover gt 50 / tree and shrub lt 50',
                          'Shrubland',
                          'Shrubland evergreen',
                          'Shrubland deciduous',
                          'Grassland',
                          'Lichens and mosses',
                          'Sparse vegetation (tree/shrub/herbaceous cover) lt 15',
                          'Sparse tree lt 15',
                          'Sparse shrub lt 15',
                          'Sparse herbaceous cover lt 15',
                          'Tree cover flooded fresh or brakish water',
                          'Tree cover flooded saline water',
                          'Shrub or herbaceous cover flooded water',
                          'Urban areas',
                          'Bare areas',
                          'Consolidated bare areas',
                          'Unconsolidated bare areas',
                          'Water bodies',
                          'Permanent snow and ice')
      
      global_shortclasses <- c('Cropland rainfed',
                               'Cropland rainfed Herbaceous cover',
                               'Cropland rainfed Tree or shrub cover',
                               'Cropland irrigated or post-flooding',
                               'Mosaic cropland gt 50 natural vegetation (tree/shrub/herbaceous cover) lt 50',
                               'Mosaic natural vegetation gt 50 cropland lt 50',
                               'Tree cover broadleaved evergreen closed to open gt 15',
                               'Tree cover  broadleaved  deciduous  closed to open gt 15',
                               'Tree cover  broadleaved  deciduous  closed gt 40',
                               'Tree cover  broadleaved  deciduous  open 15 to 40',
                               'Tree cover  needleleaved  evergreen  closed to open gt 15',
                               'Tree cover  needleleaved  evergreen  closed gt 40',
                               'Tree cover  needleleaved  evergreen  open 15 to 40',
                               'Tree cover  needleleaved  deciduous  closed to open gt 15',
                               'Tree cover  needleleaved  deciduous  closed gt 40',
                               'Tree cover  needleleaved  deciduous  open 15 to 40',
                               'Tree cover  mixed leaf type',
                               'Mosaic tree and shrub gt 50 herbaceous cover lt 50',
                               'Mosaic herbaceous cover gt 50 / tree and shrub lt 50',
                               'Shrubland',
                               'Shrubland evergreen',
                               'Shrubland deciduous',
                               'Grassland',
                               'Lichens and mosses',
                               'Sparse vegetation (tree/shrub/herbaceous cover) lt 15',
                               'Sparse tree lt 15',
                               'Sparse shrub lt 15',
                               'Sparse herbaceous cover lt 15',
                               'Tree cover flooded fresh or brakish water',
                               'Tree cover flooded saline water',
                               'Shrub or herbaceous cover flooded water',
                               'Urban areas',
                               'Bare areas',
                               'Consolidated bare areas',
                               'Unconsolidated bare areas',
                               'Water bodies',
                               'Permanent snow and ice')
      
      global_colors = c('#ffff64','#ffff64','#ffff00',
                        '#aaf0f0','#dcf064','#c8c864',
                        '#006400','#00a000','#00a000',
                        '#aac800','#003c00','#003c00',
                        '#005000','#285000','#285000',
                        '#286400','#788200','#8ca000',
                        '#be9600','#966400','#966400',
                        '#be9600','#ffb432','#ffdcd2',
                        '#ffebaf','#ffc864','#ffd278',
                        '#ffebaf','#00785a','#009678',
                        '#00dc82','#c31400','#fff5d7',
                        '#dcdcdc','#fff5d7','#0046c8',
                        '#ffffff')
    }   
    
  } else{
    
    my_rules <- c(1, 8, 1,  8, 10, 2,11,12,3,13,14,3,10,11,4,14,15,4,16,17,4,12,13,5,15,16,6)
    reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
    serie <- reclassify(serie_raw,reclmat)
    
    global_classes <- c("Forest","Grassland","Cropland","Wetland","Settlement","Otherland")
    
    global_shortclasses <- c("Forest","Cropland","Grassland","Wetland","Settlement","Otherland")
    
    global_colors = c("#0b8706", "#e56e0b", "#d80be5", "#0be593", "#f60505", "#f60505")
    
  }
  
  ####################################################################################???
  ## Elaboration of a table which describes for each pixel the land category at each time step + value of covariates
  
  ## using rasterToPoints(r) to get coordinates
  tab <- as.data.frame(rasterToPoints(serie_raw))
  
  coodsls <- c("x","y")
  
  names(tab) <- c(coodsls, yearls)
  
  # A look at the first rows of the table:
  head(tab)
  # and a summary:
  summary(tab)
  
  # Put an extra column with the concatened sequence
  tab2 <- unite(tab, sec, yearls, sep="-")
  tab$sec <- tab2$sec
  
  # Verify all the categories for all the time steps
  clases <- c()
  for (year in 3:(length(yearls)+2)){
    clases <- unique(c(clases,tab[,year]))
  }
  
  ### Determine the number of different sequences
  unique_traj <- sort(unique(tab$sec))
  length(unique_traj)  # 1651 different trajectories
  
  # Determine the frequency of these trajectories
  tab_freq <- as.data.frame(table(tab$sec))
  sum(tab_freq$Freq)
  # Sort the sequences from more to less frequent
  tab_freq2 <- tab_freq[order(tab_freq$Freq,decreasing = TRUE),] 
  head(tab_freq2)
  
  return(list(global_classes, global_shortclasses, global_colors, clases, tab))
}

load_prob<- function(dir, tyear){
  files_prob <- list.files(path=dir, pattern="*.asc$", all.files=FALSE, full.names=TRUE,recursive=TRUE)
  
  files_prob <- Filter(function(x) grepl(pattern, x), files_prob)
  prob_terrai <- raster::stack(files_prob)
  return(prob_terrai)
}

## State sequence object

# dataset
tile <- 'tile'
dataset <- 'Copernicusnew_cebf'
trat <- 'post'
ckpt <- '24257'
tiles <- 1

scheme <- 'raw'
lucscheme <- 'copernicus' #esa copernicus MCD12Q1v6LCProp1 MCD12Q1v6LCProp2 MCD12Q1v6LCType1
target_year <-c(2007,2007)
yearls <- paste0("c",as.character(seq(2004,2017,1)))

if (trat == 'post'){
  # post filest
  dir <- paste0("E:/acocac/research/",tile,"/post_fc70")
  pattern <- paste0(dataset,'_afterpost_convgru15_ckp',ckpt,'_Forest90col') #MCD12Q1v6raw_LCProp1  ESAraw
} else{
  # raw files
  dir <- paste0("E:/acocac/research/",tile,"/eval/pred/7_dataset/ep5/convgru/convgru64_15_fold0_",dataset,"_",ckpt)
  pattern <- "prediction"
}

results <- calc_seq(scheme, lucscheme, target_year, yearls)  

#combined_raster <- mosaic(prob_terrai_t1_tar, prob_terrai_t2_tar, fun = sum)

global_classes = results[1][[1]]
global_shortclasses =  results[2][[1]]
global_colors =  results[3][[1]]
classes = results[4][[1]]
tab = results[5][[1]]

target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

labels <- target_classes                             
short_labels <- target_short
palette <- target_colors

tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                  cpal = palette, labels = short_labels)

# Plot the 10 most frequent sequences.
seqfplot(tab.seq, with.legend = F, main=paste0("Most common sequences for det in ",paste(target_year,collapse=" to ")), legend.prop=0.1)

###temp
seq_target <-tab.seq
df_target <- tab

## LCP
dist.lcp <- seqdist(seq_target, method = "LCP") 

## Cluster based on OM transition rates
k <- 5

## Cluster based on LCP
clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
cl_lcp <- cutree(clusterward_lcp, k = k)
df_target$clusterlcp <- cl_lcp

seqIplot(seq_target, group = df_target$clusterlcp, sortv = "from.start", legend=FALSE)

# Elaborate raster LCP
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterlcp))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcp <- raster(xyz)
crs(raster_clcp) <- "+proj=longlat +datum=WGS84 +no_defs"
plot(raster_clcp)

#prob
if (tiles > 1){
  dir_prob_tile1 <- 'E:/acocac/research/tile2/terrai/ascii/h12v10/detection_2004_01_01_to_2011_06_10'
  dir_prob_tile2 <- 'E:/acocac/research/tile2/terrai/ascii/h12v09/detection_2004_01_01_to_2012_02_02'
  
  pattern <- '2004'
  
  prob_terrai_t1 <- load_prob(dir_prob_tile1, pattern)
  prob_terrai_t2 <- load_prob(dir_prob_tile2, pattern)
  
  len_t1 <- length(names(prob_terrai_t1))
  len_t2 <- length(names(prob_terrai_t2))
  
  if (len_t1 != len_t2){
    min_ts <- pmin(len_t1, len_t2)
    prob_terrai_t1_tar <- prob_terrai_t1[[1:min_ts]]
    prob_terrai_t2_tar <- prob_terrai_t2[[1:min_ts]]
    
  }
  
  x <- list(prob_terrai_t2_tar, prob_terrai_t1_tar)
  x$filename <- 'test.tif'
  x$overwrite <- TRUE
  m <- do.call(merge, x)
  
  names(m) <- names(prob_terrai_t1_tar)
}

dates <- 'detection_2004_01_01_to_2011_06_10'
modis <- 'h13v10'
dir_prob_tile <- paste0('E:/acocac/research/',tile,'/terrai/ascii/',modis,'/dates/',dates)

pattern <- '2004'

m <- load_prob(dir_prob_tile, pattern)

#get common extent
prob_terrai_r <- m

#raster with dimensions evaluation block
xmin <- bbox(prob_terrai_r)[1,1]
xmax <- bbox(prob_terrai_r)[1,2]
ymin <- bbox(prob_terrai_r)[2,1] 
ymax <- bbox(prob_terrai_r)[2,2] 
newextent=c(xmin, xmax, ymin, ymax)

e <- as(extent(newextent), 'SpatialPolygons')
crs(e) <- "+proj=longlat +datum=WGS84 +no_defs"

rasternew <- extend(raster_clcp, e, value=NA) #fill NA
rasternew <- crop(rasternew, newextent)

#create folder
dir.create(file.path(dirname(dir), 'clusters'), showWarnings = FALSE)

writeRaster(rasternew,paste0(dirname(dir),'/clusters/','clcp1_',dataset,'_',as.character(paste(target_year,collapse="-")),'_',scheme,'_c',k,'.asc'), datatype='INT4S', overwrite=TRUE)

##export target cluster
cluster_unchanged <- 2
target.values <- function(x) { x[x!=cluster_unchanged] <- NA; return(x) }
cluster_target <- calc(raster_clcp, target.values)

#get common extent
prob_terrai_r <- m

xmin <- max(bbox(prob_terrai_r)[1,1], bbox(cluster_target)[1,1])
xmax <- min(bbox(prob_terrai_r)[1,2], bbox(cluster_target)[1,2])  
ymin <- max(bbox(prob_terrai_r)[2,1], bbox(cluster_target)[2,1])  
ymax <- min(bbox(prob_terrai_r)[2,2], bbox(cluster_target)[2,2]) 
newextent=c(xmin, xmax, ymin, ymax)

prob_terrai_sam = crop(prob_terrai_r, newextent)
cluster_target_sam = crop(cluster_target, newextent)

prob_terrai_target <- mask(x=prob_terrai_sam, mask=cluster_target_sam)
names(prob_terrai_target) <- gsub("unknown_detection_2004_01_01_to_2012_02_02_","", names(prob_terrai_target))

#boxplot(prob_terrai_target, xaxt="n")
#boxplot(prob_terrai_target)

#boxplot
dat <- as.data.frame(values(prob_terrai_target))
dat$id = rownames(dat)
dat.m <- melt(dat,id.vars='id')
dat.m$Date <- factor(as.Date(dat.m$variable, "X%Y_%m_%d"))

#data preparation
dat.m$Date <- as.POSIXct(dat.m$Date)]
dat.m$Date <- as.POSIXct(dat.m$Date)

Temperatures$Temperature <- as.numeric(Temperatures$Temperature)


p <-ggplot(data = dat.m) + 
  geom_boxplot(aes(y = value, x = variable, group = date)) +
  theme_classic() 

p <-ggplot(data = dat.m) + 
  geom_boxplot(aes(y = value, x = variable)) +
  theme_classic() 

p + theme(axis.title.x=element_blank(),
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank())


