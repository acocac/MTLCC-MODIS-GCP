##### Analysis of High Temporal Resolution Land Use/Land Cover Trajectories
##### Supplementary material to article published in Land https://www.mdpi.com/journal/land
##### R script to carry out the analysis presented in the paper
##### JF Mas - Universidad Nacional Autónoma de México - jfmas@ciga.unam.mx

# Working directory and libraries
library(TraMineR)
library(raster)
library(tidyr)
library(reshape2)
library(sampling)

set.seed(1)
##aux functions##
stratified <- function(df, group, size) {
  # USE: * Specify your data frame and grouping variable (as column
  # number) as the first two arguments.
  # * Decide on your sample size. For a sample proportional to the
  # population, enter "size" as a decimal. For an equal number
  # of samples from each group, enter "size" as a whole number.
  #
  # Example 1: Sample 10% of each group from a data frame named "z",
  # where the grouping variable is the fourth variable, use:
  #
  # > stratified(z, 4, .1)
  #
  # Example 2: Sample 5 observations from each group from a data frame
  # named "z"; grouping variable is the third variable:
  #
  # > stratified(z, 3, 5)
  #
  temp = df[order(df[group]),]
  colsToReturn <- ncol(df)
  
  #Don't want to attempt to sample more than possible
  dfCounts <- table(droplevels(df[group]))
  if (size > min(dfCounts)) {
    size <- min(dfCounts)
  }
  
  if (size < 1) {
    size = ceiling(table(droplevels(temp[group])) * size)
  } else if (size >= 1) {
    size = rep(size, times=length(table(temp[group])))
  }
  strat = strata(temp, stratanames = names(temp[group]),
                 size = size, method = "srswor")
  (dsample = getdata(temp, strat))
  
  dsample <- dsample[order(dsample[1]),]
  dsample <- data.frame(dsample[,1:colsToReturn], row.names=NULL)
  return(dsample)
  
}

mcr <- function(x, drop = FALSE) { #'most common row'
  xx <- do.call("paste", c(data.frame(x), sep = "-"))
  tx <- table(xx)
  mx <- names(tx)[which(tx == max(tx))[1]]
  return(mx)
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

samplingfreq <- function(seqdf, seqobj, ratio) { #'most common row'
  
  #compute and extract freqs
  seqdata <- seqobj[rowSums(seqobj != attr(seqobj, "nr")) != 
                                 0, ]
  weights <- attr(seqdata, "weights")
  if (is.null(weights)) {
    weights <- rep(1, nrow(seqdata))
  }
  
  seqlist <- seqconc(seqobj)
  Freq <- tapply(weights, seqlist, sum)
  Freq <- sort(Freq, decreasing = TRUE)
  
  qs = quantile(Freq,c(0.95))
  Freq2 = Freq[Freq>=qs]
  target_sampling = names(Freq2)
  
  #merge target df
  mergexyseq <- cbind(seqdf[,1:2], seqobj, seqlist)

  res <- mergexyseq[match(seqlist, names(Freq2)), ]
  res <- res[complete.cases(res), ]

  sample.tmp <- stratified(res,"Sequence",ratio)
  sample.tmp$Sequence <- factor(sample.tmp$Sequence)

  return(sample.tmp)
}
  
##seq functions##
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
    
    #cat(dim(terrai_target))
    #write terra-i raster
    #writeRaster(terrai_target, filename_raster=paste0(sample_dir,"/terrai_",Syear_terrai,"to",Eyear,".tif"), format="GTiff", overwrite=T)

    serie_raw <- mask(x=lucserie, mask=terrai_target)
  
    if (scheme == 'raw'){
      serie <- serie_raw
      if (dataset == 'MCD12Q1v6raw_LCType1') {
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

split_seq <- function(dataset, results_df, ratio){
  
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
  
  #create stratas
  std.df <- apply(tab[,3:(dim(tab)[2]-1)], 1, sd) 
  
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]

  tab.stable.seq <- seqdef(tab.stable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                           cpal = palette, labels = short_labels)
  
  sample.stable <- samplingfreq(tab.stable, tab.stable.seq, ratio)
  
  if (dim(tab.nonstable)[1] > 0){
    tab.nonstable.seq <- seqdef(tab.nonstable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                           cpal = palette, labels = short_labels)
  
    sample.nonstable <- samplingfreq(tab.nonstable, tab.nonstable.seq, ratio)
  
    outputs = list('stable'=sample.stable,'nonstable'=sample.nonstable, tyear)
  } else{
    outputs = list('stable'=sample.stable)
  }
  
  return(outputs)
}

## State sequence object

# dataset
#tiles <- c('tile_0_201')
tiles <- c('tile_0_201','tile_0_143', 'tile_1_463', 'tile_1_438', 'tile_0_630', 'tile_1_713','tile_0_365')
#datasets <- c('MCD12Q1v6raw_LCType1')
datasets <- c('MCD12Q1v6raw_LCType1','mapbiomas')
ratio <- 0.01
lc_target_years <-c(2004,2017)
terrai_dataset <- '2004_01_01_to_2019_06_10'
terrai_target_years <-c(2004,2014)

sample_dir <- paste0("E:/acocac/research/AMZ/verification/sampling/raw")
dir.create(sample_dir, showWarnings = FALSE, recursive = T)

for (tile in tiles){
  
  for (d in datasets){
    
    if (d == 'mapbiomas'){
      trat <- d
    } else{
      trat <- 'prediction' #prediction mapbiomas
      ckpt <- '42497'
      tiles <- 1
      epochs <- 30
    }
    
    if (d %in% c('mapbiomas')){
      schemes <- 'raw' #raw
    } else {
      schemes <- c('raw','simple') #simple
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
      output <- split_seq(d, results, ratio)
      
      if (len(output) > 2){
        
          targetyear = output[[3]]
          
          print(paste0('Tile ',tile))
          print(paste0('Dataset ',d,' & Scheme ',scheme,' & Tyear ',targetyear))
          print(paste0('size stable = ',dim(output[[1]])[1]))
          print(paste0('size non-stable = ',dim(output[[2]])[1]))
          
          write.csv(cbind(tile=tile,output[[1]]),file=paste0(sample_dir,"/stable_",d,"_",scheme,"_",tile,"def",targetyear,"_sratio",gsub("[.]","_",ratio),".csv"), row.names=F)
          write.csv(cbind(tile=tile,output[[2]]),file=paste0(sample_dir,"/nonstable_",d,"_",scheme,"_",tile,"def",targetyear,"_sratio",gsub("[.]","_",ratio),".csv"), row.names=F)
          
      } else{
        
          targetyear = output[[2]]
          
          print(paste0('Tile ',tile))
          print(paste0('Dataset ',d,' & Scheme ',scheme,' & Tyear ',targetyear))
          print(paste0('size stable = ',dim(output[[1]])[1]))
  
          write.csv(cbind(tile=tile,output[[1]]),file=paste0(sample_dir,"/stable_",d,"_",scheme,"_",tile,"def",targetyear,"_sratio",gsub("[.]","_",ratio),".csv"), row.names=F)
      }

    }
  }
}