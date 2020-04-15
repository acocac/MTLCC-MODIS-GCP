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
## State sequence object

# dataset
tile <- 'tile2'
dataset <- 'MCD12Q1v6raw_LCProp2'
trat <- 'post'

if (scheme == 'post'){
  # post filest
  dir <- paste0("E:/acocac/research/",tile,"/post_fc70")
  pattern <- paste0(dataset,'_afterpost_convgru15_ckp19393_Forest90col') #MCD12Q1v6raw_LCProp1  ESAraw
} else{
  # raw files
  dir <- paste0("E:/acocac/research/",tile,"/eval/pred/7_dataset/ep5/convgru/convgru64_15_fold0_",dataset,"_19393")
  pattern <- "prediction"
}

scheme <- 'raw'
lucscheme <- 'MCD12Q1v6LCProp2' #esa copernicus MCD12Q1v6LCProp1 MCD12Q1v6LCProp2 MCD12Q1v6LCType1
target_year <-c(2006,2006)
yearls <- paste0("c",as.character(seq(2004,2017,1)))

results <- calc_seq(scheme, lucscheme, target_year, yearls)  

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

#######################################################################################
## Visualize the sequence data set 

# Plot 10 sequences in the tab.seq sequence object (chosen to show diversity)
# some_seq <- tab.seq[c(19,5,973,976,34,84,930,3893,993,995),]
# seqiplot(some_seq, with.legend = T, border = T, main = "Some sequences", legend.prop=0.2)

# Plot all the sequences in the data set, sorted by states from start.
seqIplot(tab.seq, sortv = "from.start", with.legend = F, main = paste0("Sequences 2004-2017 for det in ",paste(target_year,collapse=" to ")))

# Plot the 10 most frequent sequences.
seqfplot(tab.seq, with.legend = F, main=paste0("Most common sequences for det in ",paste(target_year,collapse=" to ")), legend.prop=0.1)

#######################################################################################
## Explore the sequence data set by computing and visualizing descriptive statistics

# Compute and plot the state distributions by time step. 
# With border = NA, borders surrounding the bars are removed. 
seqdplot(tab.seq, with.legend = T, border = NA,main="Land cover (states) distribution", ylab="Proportion of study area", legend.prop=0)

# Compute and plot the transversal entropy index (Landscape entropy over time)
seqHtplot(tab.seq, main = "Entropy", ylab="Entropy index value",xlab=("Time"), legend.prop=0.2)

#Plot the sequence of modal states (dominant land cover) of the transversal state distributions.
seqmsplot(tab.seq, with.legend = T, main ="Most frequent land cover", legend.prop=0.2)

# Plot the mean time spent in each land cover category.
seqmtplot(tab.seq, with.legend = T, main = "Permanence", ylab="Number of 3 years periods", legend.prop=0.3)

### Longitudinal turbulence and entropy indices 
# Computed for each pixel over the time
tab$Turb <- seqST(tab.seq, norm=FALSE)
tab$Entrop <- seqient(tab.seq, norm=TRUE, base=exp(1))

# Generate rasters which represent these indices
xyt <- as.data.frame(cbind(tab$x,tab$y,tab$Turb))
names(xyt) <- c("x","y","t")
coordinates(xyt) <- ~ x + y
gridded(xyt) <- TRUE
raster_turb <- raster(xyt)
plot(raster_turb)

xye <- as.data.frame(cbind(tab$x,tab$y,tab$Entrop))
names(xye) <- c("x","y","e")
head(xye)
coordinates(xye) <- ~ x + y
gridded(xye) <- TRUE
raster_entrop <- raster(xye)
plot(raster_entrop)

# Calculate the correlation between both indices
cor(tab$Turb,tab$Entrop) # 0.9394874

## Computes the transition rates
tr_rates <- seqtrate(tab.seq)
print(tr_rates)

#######################################################################################
# Compute distances between sequences using different dissimilarity indices

## OM with substitution costs based on transition
## probabilities and indel set as half the maximum
## substitution cost
costs.tr <- seqcost(tab.seq, method = "TRATE",with.missing = FALSE)

seq_target <-tab.seq
df_target <- tab

dist.om1 <- seqdist(seq_target, method = "OM",indel = costs.tr$indel, sm = costs.tr$sm,with.missing = F)
dim(dist.om1)

# ### OM based on features (10 to Forest, 5 to savanna, 3 for pasture, agriculture and mosaic and 1 for others 
# globa_state_features <- c(10,10,10,10,8,8,8,8,5,5,5,5,1,1,1,1,1)
# targetstate_features <- globa_state_features[sort(classes)]
# 
# tab_state_features <- data.frame(state=targetstate_features)
# costs.gower <- seqcost(seq_target, method = "FEATURES",with.missing = FALSE,state.features = tab_state_features)
# dist.om2 <- seqdist(tab.seq, method = "OM",indel = costs.gower$indel, sm = costs.gower$sm,with.missing = F)
# dim(dist.om2)

## LCS
dist.lcs <- seqdist(seq_target, method = "LCS")

## LCP
dist.lcp <- seqdist(seq_target, method = "LCP") 

# Elaboration a typology of the trajectories: build a Ward hierarchical clustering
# of the sequences from the different distances and retrieve for each cell sequence the
# cluster membership of the 5 class solution. 

## Cluster based on OM transition rates
k <- 5

clusterward_om1 <- hclust(as.dist(dist.om1),method="ward.D")
plot(clusterward_om1, xlab="", sub="", labels=FALSE)
cl_om1 <- cutree(clusterward_om1, k = k)
df_target$clusterom1 <- cl_om1
head(df_target)

## Cluster based on OM features
# clusterward_om2 <- hclust(as.dist(dist.om2),method="ward.D")
# plot(clusterward_om2)
# cl_om2 <- cutree(clusterward_om2, k = k)
# df_target$clusterom2 <- cl_om2
# head(df_target)

## Cluster based on LCS
clusterward_lcs <- hclust(as.dist(dist.lcs),method="ward.D")
plot(clusterward_lcs, xlab="", sub="", labels=FALSE)
cl_lcs <- cutree(clusterward_lcs, k = k)
df_target$clusterlcs <- cl_lcs
head(df_target)

## Cluster based on LCP
clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
plot(clusterward_lcp, xlab="", sub="", labels=FALSE)
cl_lcp <- cutree(clusterward_lcp, k = k)
df_target$clusterlcp <- cl_lcp
head(df_target)

# Plot all the sequences within each cluster para los 4 métodos
# OM1
seqIplot(seq_target, group = df_target$clusterom1, sortv = "from.start",legend=FALSE)
# OM2
# seqIplot(seq_target, group = df_target$clusterom2, sortv = "from.start")
# LCS
seqIplot(seq_target, group = df_target$clusterlcs, sortv = "from.start",legend=FALSE)
# LCP
seqIplot(seq_target, group = df_target$clusterlcp, sortv = "from.start", legend=FALSE)


######### Plot clusters' spatial distribution

# Elaborate raster OM1
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterom1))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_com1 <- raster(xyz)
plot(raster_com1)

# Elaborate raster OM2
# xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterom2))
# names(xyz) <- c("x","y","z")
# xyz <- xyz[complete.cases(xyz), ]
# coordinates(xyz) <- ~ x + y
# gridded(xyz) <- TRUE
# raster_com2 <- raster(xyz)
# plot(raster_com2)

# Elaborate raster LCS
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterlcs))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcs <- raster(xyz)
plot(raster_clcs)

# Elaborate raster LCP
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterlcp))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcp <- raster(xyz)
plot(raster_clcp)

#writeRaster(raster_clcp,paste0(dirname(dir),'/clusters/','clcp',pattern,'_',as.character(paste(target_year,collapse="-")),'_',scheme,'_c',k,'.tif'))
#writeRaster(raster_clcp,paste0(dirname(dir),'/clusters/','clcp',dataset,'_',pattern,'_',as.character(paste(target_year,collapse="-")),'_',scheme,'_c',k,'.tif'))

# #######################################################################################
# ## Run discrepancy analyses to study how sequences are related to covariates
# 
# # Compute and test the share of discrepancy explained by different categories on covariates 
# da1 <- dissassoc(dist.om1, group = tab$slopeK, R = 50)
# print(da1$stat)
# da2 <- dissassoc(dist.om1, group = tab$distK, R = 50)
# print(da2$stat)
# da3 <- dissassoc(dist.om1, group = tab$distK2, R = 50)
# print(da3$stat)
# 
# 
# # Selecting event subsequences:
# # The analysis was restricted to sequences that exhibit the state Mosaic
# 
# tabe.seq <- seqecreate(tab.seq, use.labels = FALSE)
# mosaic <- seqecontain(tabe.seq, event.list = c("Mosaic"))
# mosaic_tab <- tab[mosaic,]
# mosaic.seq <- tab.seq <- seqdef(mosaic_tab, 3:13, alphabet = alphabet, states = short_labels,
#                                 cpal = palette, labels = labels)
# mosaic.seqe <- seqecreate(mosaic.seq, use.labels = FALSE)
# 
# # Look for frequent event subsequences and plot the 10 most frequent ones.
# fsubseq <- seqefsub(mosaic.seqe, pmin.support = 0.05)
# head(fsubseq)
# # 10 Most common subsequences
# plot(fsubseq[1:10], col = "grey98")
# 
# # Determine the subsequences of transitions which best discriminate the groups as
# # areas close and faraway from roads
# discr1 <- seqecmpgroup(fsubseq, group = mosaic_tab$distK2)
# plot(discr1[1:10],cex=1,cex.legend=1,legend.title="Distance",cex.lab=0.8, cex.axis = 0.8)
# # areas with moderate vs steep slope
# discr2 <- seqecmpgroup(fsubseq, group = mosaic_tab$slopeK2)
# plot(discr2[1:10],cex=1,cex.legend=1,legend.title="Slope",cex.lab=0.8, cex.axis = 0.8)
# # clusters of sequences
# discr3 <- seqecmpgroup(fsubseq, group = mosaic_tab$clusterom1)
# plot(discr3[1:10],cex=1,cex.legend=1,legend.title="Clusters OM1",cex.lab=0.8, cex.axis = 0.8)

