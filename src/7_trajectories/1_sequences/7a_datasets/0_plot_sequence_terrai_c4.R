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
library(TraMineRextras)

##Function Calculate Statistics##
calc_seq <- function(terrai_dataset, scheme, dataset, terrai_target, lc_target){
    
    # scheme <- 'raw'
    # target_year <-c(2004,2014)
    # yearls <- paste0("c",as.character(seq(2004,2017,1)))
    
    files <- list.files(path=dir, pattern="*.tif$", all.files=FALSE, full.names=TRUE,recursive=TRUE)
    
    #subset dataset
    files <- Filter(function(x) grepl(pattern, x), files)

        #subset years
    wildcard <- paste0(as.character(seq(lc_target[1],lc_target[2],1)),collapse="|")
    files <- Filter(function(x) grepl(wildcard, x), files)
  
    lucserie <- raster::stack(files)
    # plot(lucserie[[1]])
    
    watermask_file <- dir <- paste0("E:/acocac/research/",tile,"/eval/verification/watermask/2018/watermask/0_0_0.tif")
    watermask_raster <- raster(watermask_file)
    
    terrai_file <- paste0('T:/GISDATA_terra/outputs/Latin/',terrai_dataset,'/TIFF/GEOGRAPHIC/WGS84/decrease/region/classified/latin_decrease_',terrai_dataset,'.tif')
    terrai_raster <- raster(terrai_file)
    e <- extent(lucserie)
    
    terrai.crop <- crop(terrai_raster, e)
    watermask.crop <- crop(watermask_raster, e)

    terrai.masked = terrai.crop * watermask.crop
    
    #added to compute year with maximum
    freq.tb = freq(terrai.masked, useNA='no')
    freq.tb = freq.tb[which(freq.tb[,1] == terrai_target[1]):which(freq.tb[,1] == terrai_target[2]),]
    
    tyear = freq.tb[which.max(freq.tb[,2])]
    
    # plot(terrai.crop)
    
    #Syear <- terrai_target[1] old multiple periods
    #Eyear <- terrai_target[2] old multiple periods
    
    Syear <- tyear
    Eyear <- tyear
                           
    target.values <- function(x) { x[x<Syear] <- NA; return(x) }
    terrai_target <- calc(terrai.masked, target.values)
    target.values <- function(x) { x[x>Eyear] <- NA; return(x) }
    terrai_target <- calc(terrai_target, target.values)
    target.values <- function(x) { x[!is.na(x)] <- 1; return(x) }
    terrai_target <- calc(terrai_target, target.values)
    
    png(filename=paste0(chart_dir,"/terrai_",Syear,"to",Eyear,".png"))
    plot(terrai_target, legend=FALSE, col=gray(0:100/100))
    dev.off()
    
    serie_raw <- mask(x=lucserie, mask=terrai_target)

    if (scheme == 'raw'){
      serie <- serie_raw
      if (dataset == 'Copernicusraw') {
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
      else if (dataset == 'MCD12Q1v6raw_LCType1') {
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
                          '#fa0000',
                          '#ff6d4c',
                          '#69fff8',
                          '#f9ffa4',
                          '#1c0dff')
      }
      else if (dataset == 'MCD12Q1v6raw_LCProp1') {
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
      else if (dataset == 'MCD12Q1v6raw_LCProp2') {
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
                          '#fa0000','#003f00','#006c00',
                          '#e3ff77','#b6ff05','#93ce04',
                          '#77a703','#dcd159')
      } 
      else if (dataset == 'ESAraw') {
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
        
        global_shortclasses <- c('CR',
                                 'CH',
                                 'CT',
                                 'CI',
                                 'MCN50',
                                 'MNC50',
                                 'TBE',
                                 'TBD',
                                 'TBDC',
                                 'TBDO',
                                 'TNE',
                                 'TNEC',
                                 'TNEO',
                                 'TND',
                                 'TNDC',
                                 'TNDO',
                                 'TMX',
                                 'MTH50',
                                 'MHT50',
                                 'S',
                                 'Se',
                                 'Sd',
                                 'G',
                                 'LM',
                                 'SV',
                                 'St',
                                 'Ss',
                                 'Sh',
                                 'TFF',
                                 'TFS',
                                 'SHF',
                                 'Bu',
                                 'Ba',
                                 'Bac',
                                 'Bau',
                                 'W',
                                 'SI')
        
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
      else if (dataset == 'Copernicusnew_cf2others') {
        global_classes <- c('Dense forest',
                            'Open forest',
                            'Shrubs',
                            'Herbaceous vegetation',
                            'Bare - sparse vegetation',
                            'Urban - built up',
                            'Cropland',
                            'Water',
                            'Herbaceous wetland')
        
        global_shortclasses <- c('DF', 'OF',
                                 'S', 'HV',
                                 'Ba', 'Bu', 'C',
                                 'W', 'HW')
        
        global_colors = c('#086a10',
                          '#54a708', 
                          '#ffbb22',
                          '#ffff4c',
                          '#b4b4b4',
                          '#fa0000',
                          '#f096ff',
                          '#0032c8',
                          '#0096a0')
        #ffbb22,#ffff4c,#0096a0,#b4b4b4,#f096ff,#fa0000,#0032c8
      }   
      else if (dataset == 'merge_datasets2own') {
        global_classes <- c('Barren',
                            'Water Bodies',
                            'Urban and Built-up Lands',
                            'Dense Forests',
                            'Open Forests',
                            'Natural Herbaceous',
                            'Croplands',
                            'Shrublands')
        
        global_shortclasses <- c('Ba', 
                                 'W', 'Bu',
                                 'DF', 'OF', 
                                 'NH', 'C',
                                 'S')
        
        global_colors = c('#f9ffa4','#1c0dff',
                          '#fa0000','#003f00',
                          '#006c00','#b6ff05',
                          '#77a703','#dcd159')
      }  
      else if (dataset == 'mapbiomas') {
        global_classes <- c('Forest Formation', 'Savanna Formation',
                            'Mangrove', 'Flooded forest',
                            'Wetland', 'Grassland', 'Other non forest natural formation',
                            'Farming', 'Non vegetated area', 'Salt flat', 'River, Lake and Ocean',
                            'Glacier')
        
        global_shortclasses <- c('F','S', 'M',
                                 'Ff', 'We', 'G', 
                                 'NFN', 'C', 'NVA',
                                 'Sf','W','SI')
        
        global_colors = c('#009820','#00FE2D','#68743A','#74A5AF',
                          '#3CC2A6','#B9AE53','#F3C13C','#FFFEB5',
                          '#EC9999','#FD7127','#001DFC','#FFFFFF')
      }  
      
    } else if (scheme == 'simpleWwoodland'){
      if (dataset == 'MCD12Q1v6raw_LCType1') {
        
        #my_rules <- c(1,5,1,5,7,3,7,8,2,8,9,4,9,10,5,10,11,10,11,12,8,12,13,7,13,14,8,14,15,11,15,16,6,16,17,9)
        my_rules <- c(1,5,1,5,7,2,7,8,3,8,9,4,9,10,5,10,11,10,11,12,8,12,13,7,13,14,8,14,15,11,15,16,6,16,17,9)
        
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)

        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c('Forest',
                            'Shrubs',
                            'Woody savannas',
                            'Savanna',
                            'Grasslands',
                            'Barren or sparsely vegetated',
                            'Urban and built-up',
                            'Cropland',
                            'Water',
                            'Permanent wetlands',
                            'Snow and Ice')
                  
        global_shortclasses <- c('F', 'Sb',
                                 'WS', 'S',
                                 'HV', 'Ba', 'Bu', 'C',
                                 'W', 'PW','SI')
        
        global_colors = c('#086a10',
                          '#c6b044', 
                          '#dade48',
                          '#fbff13',
                          '#b6ff05',
                          '#f9ffa4',
                          '#fa0000',
                          '#c24f44',
                          '#1c0dff',
                          '#27ff87',
                          '#69fff8')
        
      }
    } else if (scheme == 'simple'){
      if (dataset == 'MCD12Q1v6raw_LCType1') {
        #my_rules <- c(1,5,1,5,7,3,7,8,2,8,9,4,9,10,5,10,11,10,11,12,8,12,13,7,13,14,8,14,15,11,15,16,6,16,17,9)
        my_rules <- c(1,5,1,5,7,2,7,8,1,8,9,3,9,10,3,10,11,8,11,12,6,12,13,5,13,14,6,14,15,9,15,16,4,16,17,7)
        
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c('Forest',
                            'Shrubs',
                            'Grasslands',
                            'Barren or sparsely vegetated',
                            'Urban and built-up',
                            'Cropland',
                            'Water',
                            'Permanent wetlands',
                            'Snow and Ice')
        
        global_shortclasses <- c('F', 'Sb', 'G', 'Ba', 'Bu', 'C',
                                 'W', 'PW','SI')
        
        global_colors = c('#086a10',
                          '#c6b044', 
                          '#fbff13',
                          '#b6ff05',
                          '#f9ffa4',
                          '#fa0000',
                          '#c24f44',
                          '#1c0dff',
                          '#27ff87',
                          '#69fff8')
      }
      
      if (dataset == 'Copernicusnew_cf2others') {
        #my_rules <- c(1,5,1,5,7,3,7,8,2,8,9,4,9,10,5,10,11,10,11,12,8,12,13,7,13,14,8,14,15,11,15,16,6,16,17,9)
        my_rules <- c(1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8,9,8)
                      
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c('Forest',
                            'Shrubs',
                            'Herbaceous vegetation',
                            'Bare - sparse vegetation',
                            'Urban - built up',
                            'Cropland',
                            'Water',
                            'Herbaceous wetland')
        
        global_shortclasses <- c('F', 
                                 'S', 'HV',
                                 'Ba', 'Bu', 'C',
                                 'W', 'HW')
        
        global_colors = c('#086a10',
                          '#ffbb22',
                          '#ffff4c',
                          '#b4b4b4',
                          '#fa0000',
                          '#f096ff',
                          '#0032c8',
                          '#0096a0')
      
      }
    }
    
    
    ####################################################################################???
    ## Elaboration of a table which describes for each pixel the land category at each time step + value of covariates
    
    #target lc data
    #yearls <- paste0("c",as.character(seq(lc_target[1],lc_target[2],1)))
    yearls <- paste0(as.character(seq(lc_target[1],lc_target[2],1)))
    
      ## using rasterToPoints(r) to get coordinates
    tab <- as.data.frame(rasterToPoints(serie))
  
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
    
    return(list(global_classes, global_shortclasses, global_colors, clases, tab, tyear))
  }

mcr <- function(x, drop = FALSE) { #'most common row'
  xx <- do.call("paste", c(data.frame(x), sep = "-"))
  tx <- table(xx)
  mx <- names(tx)[which(tx == max(tx))[1]]
  return(mx)
}

load_prob <- function(dir, tyear){
  files_prob <- list.files(path=dir, pattern="*.asc$", all.files=FALSE, full.names=TRUE,recursive=TRUE)
  
  files_prob <- Filter(function(x) grepl(pattern, x), files_prob)
  prob_terrai <- raster::stack(files_prob)
  return(prob_terrai)
}

seqfreqidx <- function(seqobj) { #'most common row'
  #determine frequency
  bf.freq <- seqtab(seqobj, idxs=nrow(seqobj))
  bf.tab <- attr(bf.freq,"freq")
  bf.perct <- bf.tab[,"Freq"]
  
  #bf.freq <- bf.freq[bf.perct > 10,]
  qs = quantile(bf.perct,c(0.95))
  bf.freq <- bf.freq[bf.perct >= qs[1],]

  if (dim(bf.freq)[1] > 1){
    (nfreq <- dim(bf.freq)[1])
  } else {
    nfreq <- 1
  }
  return(nfreq)
}

plot_seq <- function(dataset, results_df, sizew, sizeh){
  #combined_raster <- mosaic(prob_terrai_t1_tar, prob_terrai_t2_tar, fun = sum)

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
  
  #order distriplot by major
  tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                    cpal = palette, labels = short_labels)
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
                      labels = target_classes.s, cpal = target_colors.s)
  
  #create stratas
  std.df <- apply(tab[,3:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.stable.seq <- seqdef(tab.stable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                           cpal = palette, labels = short_labels)
  
  sample.stable <- samplingfreq(tab.stable, tab.stable.seq, ratio)

  if (plot == TRUE){
    sizew <- 1324
    sizeh <- 500
    if (dataset != 'mapbiomas'){
      png(filename=paste0(chart_dir,"/seqoutputs_convgru64_15_fold0_",dataset,"_",ckpt,"_",trat,"_",scheme,"_def",tyear,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
    } else{
      png(filename=paste0(chart_dir,"/seqoutputs_",dataset,"_def",tyear,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
    }
  }
  
  if (dim(tab.nonstable)[1] > 0){
    tab.nonstable.seq <- seqdef(tab.nonstable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                                cpal = palette, labels = short_labels)
    
    if (plot == TRUE){
      layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
      par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(0.8, 0.6, 0.8, 0.6))
      seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=2.4)
      seqfplot(tab.stable.seq, idxs=1:nrow(tab.stable.seq), with.legend = F, border = NA, main="Stable post-loss LC sequences \n (All)", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)
      seqfplot(tab.nonstable.seq, idxs=1:seqfreqidx(tab.nonstable.seq), with.legend = F, border = NA, main="Non-stable post-loss LC sequences \n (95% quantile)", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)
      dev.off()
    }
    
    sample.nonstable <- samplingfreq(tab.nonstable, tab.nonstable.seq, ratio)
    
    outputs = list('stable'=sample.stable,'nonstable'=sample.nonstable, 'tyear'=tyear)
    
  } else {
    
    if (plot == TRUE){
    layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
    par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(0.8, 0.6, 0.8, 0.6))
    seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=2.4)
    seqfplot(tab.stable.seq, idxs=1:nrow(tab.stable.seq), with.legend = F, border = NA, main="Stable post-loss LC sequences \n (All)", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)
    dev.off()
    }
    
    outputs = list('stable'=sample.stable, 'tyear'=tyear)
    
  }
  
  outputs = list('tab'=tab,'tab.seq'=tab.seq)
  return(outputs)
}

## State sequence object

# dataset
#tiles <- c('tile_0_201')
tiles <- c('tile_0_201','tile_0_143', 'tile_1_438', 'tile_0_630','tile_1_713','tile_0_365','tile_1_463')
datasets <- c('mapbiomas','merge_datasets2own', 'Copernicusraw','ESAraw', 'Copernicusnew_cf2others', 'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6raw_LCType1')
#datasets <- c('merge_datasets2own')

lc_target_years <-c(2004,2017)
terrai_dataset <- '2004_01_01_to_2019_06_10'
terrai_target_years <-c(2004,2010)

for (tile in tiles){
  
  chart_dir <- paste0("E:/acocac/research/",tile,"/post/pngs")
  dir.create(chart_dir, showWarnings = FALSE, recursive = T)
  
  for (d in datasets){
    
    if (d == 'mapbiomas'){
      trat <- d
    } else {
      trat <- 'prediction' #prediction mapbiomas
      ckpt <- '42497'
      epochs <- 30
    }

    if (d %in% c('MCD12Q1v6raw_LCType1','Copernicusnew_cf2others')){
      schemes <-c('raw','simple') #simple
    } else {
      schemes <- c('raw') #simple
    }
    
    for (scheme in schemes){
      if (trat == 'prediction'){
        # post filest
        dir <- paste0("E:/acocac/research/",tile,"/post")
        pattern <- paste0(d,'_',ckpt) #MCD12Q1v6raw_LCProp1  ESAraw
      } else if (trat == 'raw') {
        # raw files
        dir <- paste0("E:/acocac/research/",tile,"/eval/pred/1_dataset/ep",epochs,"/convgru/convgru64_15_fold0_",dataset,"_",ckpt)
        pattern <- "prediction"
      } else if (trat == 'mapbiomas') {
        dir <- paste0("E:/acocac/research/",tile,"/eval/verification/mapbiomas")
        pattern <- "mapbiomas"
      }
      
      results <- calc_seq(terrai_dataset, scheme, d, terrai_target_years, lc_target_years)  
      output <- plot_seq(d, results)
    }
  }
}