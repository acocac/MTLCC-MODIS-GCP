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
  
  terrai_file <- paste0('T:/GISDATA_terra/outputs/Latin/',terrai_dataset,'/TIFF/GEOGRAPHIC/WGS84/decrease/region/classified/latin_decrease_',terrai_dataset,'.tif')
  terrai_raster <- raster(terrai_file)
  e <- extent(lucserie)
  
  terrai.crop <- crop(terrai_raster, e)
  # plot(terrai.crop)
  
  # target_year <- year
  # target.values <- function(x) { x[x!=target_year] <- NA; return(x) }
  # terrai_target <- calc(terrai.crop, target.values)
  # 
  Syear <- terrai_target[1]
  Eyear <- terrai_target[2]
  
  if (Syear < 2004){
    Syear_terrai = 2004
  } else{
  Syear_terrai=Syear
  }
  
  target.values <- function(x) { x[x<Syear_terrai] <- NA; return(x) }
  terrai_target <- calc(terrai.crop, target.values)
  target.values <- function(x) { x[x>Eyear] <- NA; return(x) }
  terrai_target <- calc(terrai_target, target.values)
  target.values <- function(x) { x[!is.na(x)] <- 1; return(x) }
  terrai_target <- calc(terrai_target, target.values)
  
  png(filename=paste0(chart_dir,"/terrai_",Syear_terrai,"to",Eyear,".png"))
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
    
  } else{
    if (dataset == 'MCD12Q1v6LCType1') {
      my_rules <- c(1,1,1,1,1,5,5,2,2,4,6,3,8,3,9,7,10)
      reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
      serie <- reclassify(serie_raw,reclmat)
      
      global_classes <- c("Dense Forest",
                          "Open Forest",
                          'Cropland',
                          'Grassland',
                          'Shrublands',
                          'Water Bodies',
                          'Urban and Built-up Lands',
                          'Barren',
                          'Permanent Snow and Ice',
                          'Wetland')
                          
      global_shortclasses <- c("DF","OF","C","G","S","W","Bu","Ba","SI","We")
      
      global_colors = c("#086a10", "#54a708", "#f096ff", "#ffff4c", "#ffbb22", "#0032c8", "#fa0000","b4b4b4","ffffff","0096a0")
    }
      else if (dataset == 'MCD12Q1v6LCProp1') {
        my_rules <- c(8,9,6,7,1,2,3,4,3,3,5)
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c("Dense Forest",
                            "Open Forest",
                            'Dense Herbaceous',
                            'Sparse Herbaceous',
                            'Shrublands',
                            'Water Bodies',
                            'Barren',
                            'Permanent Snow and Ice')
        
        global_shortclasses <- c("DF","OF",'DH','SH','S','W','Ba','SI')
        
        global_colors = c("#086a10", "#54a708", "#00d000", "98d604", "ffbb22", "0032c8", "b4b4b4","ffffff") 
        
        
      }
      else if (dataset == 'MCD12Q1v6LCProp2') {
        my_rules <- c(7,8,6,1,1,1,1,1,1,2,2,3,4,5,5,5)
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c("Dense Forest",
                            "Open Forest",
                            'Cropland',
                            'Grassland',
                            'Shrublands',
                            'Water Bodies',
                            'Urban and Built-up Lands',
                            'Barren',
                            'Permanent Snow and Ice')

        global_shortclasses <- c("DF","OF","C","G","S","W","Bu","Ba","SI","We")
        
        global_colors = c("#086a10", "#54a708", "#f096ff", "#ffff4c", "#ffbb22", "#0032c8", "#fa0000","b4b4b4","ffffff","0096a0")
      }
      else if (dataset == 'esa') {
        my_rules <- c(3,3,3,3,3,3,1,1,1,2,1,1,2,1,1,2,1,2,4,5,5,5,4,8,8,8,8,8,10,10,10,7,8,8,8,6,9)
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c("Dense Forest",
                            "Open Forest",
                            'Cropland',
                            'Grassland',
                            'Shrublands',
                            'Water Bodies',
                            'Urban and Built-up Lands',
                            'Barren',
                            'Permanent Snow and Ice',
                            'Wetland')
        
        global_shortclasses <- c("DF","OF","C","G","S","W","Bu","Ba","SI","We")
        
        global_colors = c("#086a10", "#54a708", "#f096ff", "#ffff4c", "#ffbb22", "#0032c8", "#fa0000","b4b4b4","ffffff","0096a0")
        
      }
      else if (dataset == 'copernicus') {
        my_rules <- c(1,1,1,1,1,1,2,2,2,2,2,2,5,4,10,8,8,3,7,9,6,6)
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c("Dense Forest",
                            "Open Forest",
                            'Cropland',
                            'Grassland',
                            'Shrublands',
                            'Water Bodies',
                            'Urban and Built-up Lands',
                            'Barren',
                            'Permanent Snow and Ice',
                            'Wetland')
        
        global_shortclasses <- c("DF","OF","C","G","S","W","Bu","Ba","SI","We")
        
        global_colors = c("#086a10", "#54a708", "#f096ff", "#ffff4c", "#ffbb22", "#0032c8", "#fa0000","b4b4b4","ffffff","0096a0")
        
      }
  }
  
  
  ####################################################################################???
  ## Elaboration of a table which describes for each pixel the land category at each time step + value of covariates
  
  #target lc data
  #yearls <- paste0("c",as.character(seq(lc_target[1],lc_target[2],1)))
  yearls <- paste0(as.character(seq(lc_target[1],lc_target[2],1)))
  
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

plot_seq <- function(dataset, results_df, sizew, sizeh){
  #combined_raster <- mosaic(prob_terrai_t1_tar, prob_terrai_t2_tar, fun = sum)
  results_df=results
  global_classes = results_df[1][[1]]
  global_shortclasses =  results_df[2][[1]]
  global_colors =  results_df[3][[1]]
  classes = results_df[4][[1]]
  tab = results_df[5][[1]]
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  labels <- target_classes                             
  short_labels <- target_short
  palette <- target_colors
  
  #yearls <- paste0("c",as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  
  tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                    cpal = palette, labels = short_labels)
  
  #create seq excluding common
  tab2 = apply(tab,2,function(x)gsub('\\s+', '',x))
  tab_all <- apply(tab2[,3:dim(tab2)[2]], 1, paste , collapse="-") 
  common = mcr(tab2[,3:dim(tab2)[2]])
  tab.clean = tab2[!tab_all==common,]
  
  tab.clean.seq <- seqdef(tab.clean, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                          cpal = palette, labels = short_labels)
  
  #order distriplot by major
  ## Get state freq with seqmeant
  mt <- seqmeant(tab.seq)
  ## order of frequencies
  ord <- order(mt, decreasing = TRUE)
  
  ## Sorted alphabet
  alph.s <- rownames(mt)[ord]
  ## we need also to sort accordingly labels and colors
  labels
  target_classes.s <- target_classes[ord]
  target_short.s <- target_short[ord]
  target_colors.s <- target_colors[ord]
  
  ## Define sequence object with sorted states
  tab.seq.s <- seqdef(tab.seq, alphabet = alph.s, states = target_short.s,
                       labels = target_classes.s, cpal = target_colors.s)
  
  # default method
  # png(filename=paste0(chart_dir,"/all3in1_convgru64_15_fold0_",dataset,"_",ckpt,"_",trat,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
  # layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
  # par(mar = c(4, 5, 4, 1), cex.axis = 0.75, cex.main = 2.5, cex.lab=2, mai = c(0.8, 0.6, 0.8, 0.6))
  # seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution over time", ylab=paste0("Fraction [0-1] of Terra-i's det ",paste(terrai_target_years,collapse=" to ")), legend.prop=0.2, cex.axis=2.2)
  # seqfplot(tab.seq, with.legend = F, border = NA, main="Most common LC sequences \n incl. dominant", legend.prop=0.2, cex.axis=2.2)
  # seqfplot(tab.clean.seq, with.legend = F, border = NA, main="Most common LC sequences \n excl. dominant", legend.prop=0.2, cex.axis=2.2)
  # dev.off()
  
  # show 50% based on https://stackoverflow.com/questions/17444839/seqfplot-percentage-vs-number-of-most-frequent-sequences/17448424
  bf.freq <- seqtab(tab.clean.seq, idxs=nrow(tab.clean.seq))
  bf.tab <- attr(bf.freq,"freq")
  bf.perct <- bf.tab[,"Percent"]
  
  ## Compute the cumulated percentages
  bf.cumsum <- cumsum(bf.perct)
  
  bf.freq <- bf.freq[bf.cumsum <= 50,]
  
  if (dim(bf.freq)[1] > 1){
    (nfreq <- length(bf.cumsum[bf.cumsum <= 50]))
  } else {
    nfreq <- 5
  }
  
  if (dataset != 'mapbiomas'){
    png(filename=paste0(chart_dir,"/seqoutputs_convgru64_15_fold0_",dataset,"_",ckpt,"_",trat,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
  } else{
    png(filename=paste0(chart_dir,"/seqoutputs_",dataset,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
  }
  
  layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
  par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(0.8, 0.6, 0.8, 0.6))
  seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=2.4)
  seqfplot(tab.seq, with.legend = F, border = NA, main="Most common LC sequences \n incl. dominant", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)
  seqfplot(tab.clean.seq, idxs=1:nfreq, with.legend = F, border = NA, main="Most common LC sequences \n excl. dominant", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)
  dev.off()
  
  outputs = list('tab'=tab,'tab.seq'=tab.seq)
  return(outputs)
}

## State sequence object

# dataset
tile <- 'tile_0_630'
#datasets <- c('merge_datasets2own', 'Copernicusraw','ESAraw', 'Copernicusnew_cf2others', 'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6raw_LCType1')
datasets <- c('mapbiomas')


if ('mapbiomas'%in% datasets){
  trat <- 'mapbiomas'
} else{
  trat <- 'prediction' #prediction mapbiomas
  ckpt <- '42497'
  tiles <- 1
  epochs <- 30
}

lc_target_years <-c(2004,2017)
scheme <- 'raw' #simple
terrai_dataset <- '2004_01_01_to_2019_06_10'
terrai_target_years <-c(2004,2014)
chart_dir <- paste0("E:/acocac/research/",tile,"/post/pngs")
dir.create(chart_dir, showWarnings = FALSE)

for (d in datasets){
  
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
  output <- plot_seq(d, results, 1324, 500)
}

##filter sequences less <10%
results_df=results
global_classes = results_df[1][[1]]
global_shortclasses =  results_df[2][[1]]
global_colors =  results_df[3][[1]]
classes = results_df[4][[1]]
tab = results_df[5][[1]]
target_classes <- global_classes[sort(classes)]
target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]

alphabet <- sort(classes)

labels <- target_classes                             
short_labels <- target_short
palette <- target_colors

#yearls <- paste0("c",as.character(seq(lc_target_years[1],lc_target_years[2],1)))
yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))

tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                  cpal = palette, labels = short_labels)

seqfreqidx <- function(seqobj) { #'most common row'
  #determine frequency
  bf.freq <- seqtab(seqobj, idxs=nrow(seqobj))
  bf.tab <- attr(bf.freq,"freq")
  bf.perct <- bf.tab[,"Percent"]

  #bf.freq <- bf.freq[bf.perct > 10,]
  qs = quantile(bf.perct)
  cat(dim(qs))
  bf.freq <- bf.freq[bf.perct >= qs[4],]
  
  print(dim(bf.freq)[1])
  if (dim(bf.freq)[1] > 1){
    (nfreq <- dim(bf.freq)[1])
  } else {
    nfreq <- 1
  }
  return(nfreq)
}

#determine frequency
bf.freq <- seqtab(tab.stable.seq, idxs=nrow(tab.seq))
bf.tab <- attr(bf.freq,"freq")
bf.perct <- bf.tab[,"Percent"]

qs = quantile(bf.perct)
qs[3]
bf.freq <- bf.freq[bf.perct > 10,]

if (dim(bf.freq)[1] > 1){
  (nfreq <- dim(bf.freq)[1])
} else {
  nfreq <- 1
}

seqfplot(bf.freq, idxs=1:nfreq, with.legend = F, border = NA, main="Most common LC sequences \n excl. dominant", legend.prop=0.2, cex.axis=2.4, cex.lab=2.3)

#create stratas
std.df <- apply(tab[,3:(dim(tab)[2]-1)], 1, sd) 

tab.stable = tab[std.df==0,]
tab.nonstable = tab[std.df!=0,]

tab.stable.seq <- seqdef(tab.stable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                        cpal = palette, labels = short_labels)


tab.nonstable.seq <- seqdef(tab.nonstable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                         cpal = palette, labels = short_labels)

idx.stable = seqfreqidx(tab.stable.seq)
idx.nonstable = seqfreqidx(tab.nonstable.seq)

seqfplot(tab.stable.seq, idxs=1:idx.stable, with.legend = F, border = NA, main="Analysed LC stable sequences", legend.prop=0.2)
seqfplot(tab.nonstable.seq, idxs=1:idx.nonstable, with.legend = F, border = NA, main="Analysed LC non-stable sequences", legend.prop=0.2)

##after
tab <- output$tab
tab.seq <- output$tab.seq

#clusteting
###temp
df_target <- tab

#export 
tb_dir <- paste0("E:/acocac/research/",tile,"/post/tb")
dir.create(tb_dir, showWarnings = FALSE)
write.csv(df_target, paste0(tb_dir,"/sequences_convgru64_15_fold0_",dataset,"_",ckpt,"_",trat,".csv"))

## LCP
dist.lcp <- seqdist(seq_target, method = "LCP") 

## OM
costs.tr <- seqcost(tab.seq, method = "TRATE",with.missing = FALSE)
dist.om1 <- seqdist(seq_target, method = "OM",indel = costs.tr$indel, sm = costs.tr$sm,with.missing = F)

#LCS
dist.lcs <- seqdist(seq_target, method = "LCS")

#stats
dissvar.grp(dist.lcp, group=factor(seq_target[,ncol(seq_target)]))

#representatives
srep <-seqrep(seq_target, dist.matrix=dist.lcp, criterion="density",trep=0.4, tsim=0.1)
seqrplot(seq_target, dist.matrix=dist.lcp, group=factor(seq_target[,ncol(seq_target)-1]), trep=0.4, tsim=0.1, border=NA)
summary(srep)

## Computes the transition rates
tr_rates <- seqtrate(tab.seq)
print(tr_rates)

#Complexicity index
ci <- seqici(seq_target)
summary(ci)
hist(ci)

## Cluster based on OM transition rates
ctype <- 'lcp' # lcs omtr

## Cluster based on LCP
if (ctype == 'lcp'){
  clusterward <- hclust(as.dist(dist.lcp),method="ward.D")
} else if (ctype == 'lcs'){
  clusterward <- hclust(as.dist(dist.lcs),method="ward.D")
} else{
  clusterward <- hclust(as.dist(dist.om1),method="ward.D")
}
dev.off(dev.list()["RStudioGD"])

##experimental
# #cluster DBSCAN
# install.packages("fpc")
# install.packages("dbscan")
# require("dbscan")
# # db <- dbscan::dbscan(as.dist(dist.lcp), eps=0.1, minPts = 40) db <-
# db <- dbscan::dbscan(sites, eps=2, minPts = 40)
# sitesCluster <- cbind(sites,db$cluster)
# 
# db$cluster[db$cluster == 0] <- seq(max(db$cluster) + 1,
#                                    max(db$cluster) + sum(db$cluster == 0))
# seq_target$cluster_db <- db$cluster
# n_distinct(db$cluster)
# 
# pairs(a, col=a$cluster)
# 
# ##birds
# sites <- df_target
# sites$sec <-NULL
# dist_matrix <- select(sites, x, y) %>% 
#   distm %>% 
#   `/`(1000) %>% 
#   as.dist


##old plot
# plot(clusterward, xaxt='n', ann=FALSE,  axes = TRUE, labels=FALSE, cex=0.15)
# r=sort(clusterward$height,decreasing = T)
# clusterward$labels
# abline(h=r[7],col='red')

##new plot
k <- 7

#http://www.sthda.com/english/wiki/beautiful-dendrogram-visualizations-in-r-5-must-known-methods-unsupervised-machine-learning
dend <- as.dendrogram(clusterward)
nodePar <- list(lab.cex = 0.6, pch = c(NA, 19), cex = 0.7, col = "white")

plot(dend, xlab = "", sub="", ylab = "LCP distance",
     main = dataset, nodePar = nodePar, leaflab = "none")
r=sort(clusterward$height,decreasing = T)
abline(h=(r[7]+(r[7]+r[8])/2),col='red')


cl_final <- cutree(clusterward, k = k)
df_target$cl_final <- cl_final
###

#dev.off(dev.list()["RStudioGD"])
#seqIplot(seq_target, group = df_target$clusterlcp, sortv = "from.start", legend=FALSE)

# Elaborate raster LCP
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$cl_final))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcp <- raster(xyz)
crs(raster_clcp) <- "+proj=longlat +datum=WGS84 +no_defs"
dev.off(dev.list()["RStudioGD"])
plot(raster_clcp)
seqIplot(seq_target, group = df_target$cl_final, sortv = "from.start", legend=FALSE)

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
``

