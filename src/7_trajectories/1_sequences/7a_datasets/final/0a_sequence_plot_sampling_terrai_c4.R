##### Analysis of High Temporal Resolution Land Use/Land Cover Trajectories
##### Supplementary material to article published in Land https://www.mdpi.com/journal/land
##### R script to carry out the analysis presented in the paper
##### JF Mas - Universidad Nacional Autónoma de México - jfmas@ciga.unam.mx

# Working directory and libraries
library(TraMineR)
library(raster)
library(reshape2)
library(sampling)
library(tidyr)
library(plyr)
library(rlist)

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
  
  #dsample <- dsample[order(dsample[1]),]
  dsample <- data.frame(dsample[,1:colsToReturn], row.names=NULL)
  return(dsample)
  
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

seqfreqidx <- function(seqobj, quantile) { #'most common row'
  #determine frequency
  bf.freq <- seqtab(seqobj, idxs=nrow(seqobj))
  bf.tab <- attr(bf.freq,"freq")
  bf.perct <- bf.tab[,"Freq"]
  
  #bf.freq <- bf.freq[bf.perct > 10,]
  qs = quantile(bf.perct,c(quantile))
  bf.freq <- bf.freq[bf.perct >= qs[1],]
  
  if (dim(bf.freq)[1] > 1){
    (nfreq <- dim(bf.freq)[1])
  } else {
    nfreq <- 1
  }
  return(nfreq)
}

samplingfreq <- function(seqdf, seqobj, ratio, quantile) { #'most common row'
  
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
  qs = quantile(Freq,c(quantile))

  Freq2 = Freq[Freq>=qs]
  target_sampling = names(Freq2)

  #merge target df
  mergexyseq <- cbind(seqdf[,1:2], seqobj, seqlist)

  #TODO
  res <- mergexyseq[mergexyseq$Sequence %in% target_sampling, ]

  #res <- mergexyseq[match(seqlist, names(Freq2)), ]
  #res <- res[complete.cases(res), ]
  
  sample.tmp <- stratified(res,"Sequence",ratio)
  sample.tmp$Sequence <- factor(sample.tmp$Sequence)

  return(sample.tmp)
}

clumpraster <- function(r){
  #based on https://stackoverflow.com/questions/24465627/clump-raster-values-depending-on-class-attribute
  # get al unique class values in the raster
  clVal <- unique(r)

  # remove '0' (background)
  clVal <- clVal[!clVal %in% c(0,1)]
  
  # create a 1-value raster, to be filled in with NA's
  r.NA <- setValues(raster(r), 1)
  
  # set background values to NA
  r.NA[r==0]<- NA
  
  # loop over all unique class values
  for (i in clVal) {
    
    # create & fill in class raster
    r.class <- setValues(raster(r), NA)
    r.class[r == i]<- 1
    
    # clump class raster
    clp <- clump(r.class)
    
    # calculate frequency of each clump/patch
    cl.freq <- as.data.frame(freq(clp))
    
    # store clump ID's with frequency 1
    rmID <- cl.freq$value[which(cl.freq$count == 1)]
    
    # assign NA to all clumps whose ID's have frequency 1
    r.NA[clp %in% rmID] <- NA
  } 
  
  # multiply original raster by the NA raster
  r <- r * r.NA
  
    return(r)
  }

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
    
    watermask_file <- dir <- paste0("E:/acocac/research/",tile,"/eval/verification/watermask/water_mask.tif")
    watermask_raster <- raster(watermask_file)

    terrai_file <- paste0('T:/GISDATA_terra/outputs/Latin/',terrai_dataset,'/TIFF/GEOGRAPHIC/WGS84/decrease/region/classified/latin_decrease_',terrai_dataset,'.tif')
    terrai_raster <- raster(terrai_file)
    
    #temp
    lucserie <- crop(lucserie, extent(-65, -62, -10, -8))
    lucserie[lucserie <= 0] <- NA

    e <- extent(lucserie)
    
    terrai.crop <- crop(terrai_raster, e)
    watermask.crop <- crop(watermask_raster, e)
    
    terrai.masked = terrai.crop * (watermask.crop == 0)

    terrai_target_clump = clumpraster(terrai.masked)
    # terrai_target_clump = clumpraster(terrai.crop)
    
    terrai_final = terrai_target_clump
    # terrai_final = terrai.masked
    
        #added to compute year with maximum
    # freq.tb = freq.tb[which(freq.tb[,1] >= terrai_target[1]):which(freq.tb[,1] <= terrai_target[2]),]
    if (terrai_target[1] != terrai_target[2]){
      freq.tb = freq(terrai_final, useNA='no')
      freq.tb = freq.tb[which(freq.tb[,1] >= terrai_target[1]),]
      freq.tb = freq.tb[which(freq.tb[,1] <= terrai_target[2]),]
      tyear = freq.tb[which.max(freq.tb[,2])]
    } else{
      tyear = terrai_target[1]
    }
    
    # plot(terrai.crop)
    
    #Syear <- terrai_target[1] old multiple periods
    #Eyear <- terrai_target[2] old multiple periods
    
    Syear <- tyear
    Eyear <- tyear
                           
    target.values <- function(x) { x[x<Syear] <- NA; return(x) }
    terrai_target <- calc(terrai_final, target.values)
    target.values <- function(x) { x[x>Eyear] <- NA; return(x) }
    terrai_target <- calc(terrai_target, target.values)
    target.values <- function(x) { x[!is.na(x)] <- 1; return(x) }
    terrai_target <- calc(terrai_target, target.values)
    
    png(filename=paste0(chart_dir,"/terrai_",Syear,"to",Eyear,".png"), width = 1500, height = 1200, units='mm', res=50)
    plot(terrai_target, legend=FALSE, col=gray(0:100/100), cex.axis=3.5)
    dev.off()
    
    serie_raw <- mask(x=lucserie, mask=terrai_target)
    
    if (scheme == 'raw'){
      serie <- serie_raw
      if (endsWith(dataset, "LCProp2")) {
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
        global_colors = c('#f9ffa4', '#69fff8', '#1c0dff',
                          '#fa0000', '#003f00', '#006c00',
                          '#e3ff77', '#b6ff05', '#93ce04',
                          '#f096ff', '#dcd159') 

      }
      else if (dataset == 'mapbiomas') {
        global_classes <- c('Forest Formation', 'Savanna Formation',
                            'Mangrove', 'Flooded forest',
                            'Wetland', 'Grassland', 'Other non forest natural formation',
                            'Farming', 'Non vegetated area', 'Salt flat', 'River-Lake and Ocean',
                            'Glacier')
        
        global_shortclasses <- c('F','S', 'M',
                                 'Ff', 'We', 'G', 
                                 'NFN', 'C', 'NVA',
                                 'Sf','W','SI')
        
        global_colors = c('#003f00','#006c00','#68743A','#74A5AF',
                          '#3CC2A6','#b6ff05','#F3C13C','#f096ff',
                          '#f9ffa4','#FD7127','#0032c8','#FFFFFF')
      }  
    } else if (scheme == 'simple'){
      if (endsWith(dataset, "Type1")) {
        my_rules <- c(1,5,1,5,7,2,7,8,3,8,9,4,9,10,4,10,11,9,11,12,7,12,13,6,13,14,7,14,15,10,15,16,5,16,17,8)
        
        reclmat <- matrix(my_rules, ncol=3, byrow=TRUE)
        
        serie <- reclassify(serie_raw,reclmat)
        
        global_classes <- c('Forest',
                            'Shrubs',
                            'Woody savannas',
                            'Grasslands',
                            'Barren or sparsely vegetated',
                            'Urban and built-up',
                            'Cropland',
                            'Water',
                            'Permanent wetlands',
                            'Snow and Ice')
        
        global_shortclasses <- c('F', 'Sb',
                                 'WS', 'G', 'Ba', 'Bu', 'C',
                                 'W', 'PW','SI')
        
        global_colors = c('#086a10',
                          '#c6b044', 
                          '#dade48',
                          '#fbff13',
                          '#f9ffa4',
                          '#fa0000',
                          '#c24f44',
                          '#1c0dff',
                          '#27ff87',
                          '#69fff8')
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

split_seq <- function(dataset, results_df, ratio, quantile, plot=FALSE){
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
  
  if (dim(tab.stable)[1] > 0){
    tab.stable.seq <- seqdef(tab.stable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                           cpal = palette, labels = short_labels, with.missing = TRUE)
  }
  
  if (ratio > 0){
    sample.stable <- samplingfreq(tab.stable, tab.stable.seq, ratio, quantile)
  }
  
  if (plot == TRUE){
    sizew <- 1300
    sizeh <- 400
    if (dataset != 'mapbiomas'){
      png(filename=paste0(chart_dir,"/seqoutputs_convgru64_15_fold0_",dataset,"_",ckpt,"_",trat,"_",scheme,"_def",tyear,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
    } else{
      png(filename=paste0(chart_dir,"/seqoutputs_",dataset,"_def",tyear,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
    }
  }
  
  if (dim(tab.nonstable)[1] > 0){
    
    tab.nonstable.seq <- seqdef(tab.nonstable, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                                cpal = palette, labels = short_labels)
    
    if ((plot == TRUE) & (dim(tab.stable)[1] > 0)){
      layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
      par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(1, 0.6, 0.8, 0.6))
      seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=2.4)
      abline(v = pos_year, col="black", lwd=3, lty=2)
      seqfplot(tab.stable.seq, idxs=1:nrow(tab.stable.seq), with.legend = F, border = NA, main="Stable LC sequences \n (All)", legend.prop=0.2, cex.axis=2.4, cex.lab=1.9)
      abline(v = pos_year, col="black", lwd=3, lty=2)
      seqfplot(tab.nonstable.seq, idxs=1:seqfreqidx(tab.nonstable.seq, quantile), with.legend = F, border = NA, main=paste0("Non-stable LC sequences \n (",as.character(round(quantile*100)),"% quantile)"), legend.prop=0.2, cex.axis=2.4, cex.lab=1.9)
      abline(v = pos_year, col="black", lwd=3, lty=2)
      dev.off()
    }
    
    else if ((plot == TRUE) & (dim(tab.stable)[1] == 0)){
      layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
      par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(1, 0.6, 0.8, 0.6))
      seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=2.1)
      abline(v = pos_year, col="black", lwd=3, lty=2)
      seqfplot(tab.nonstable.seq, idxs=1:seqfreqidx(tab.nonstable.seq, quantile), with.legend = F, border = NA, main=paste0("Non-stable LC sequences \n (",as.character(round(quantile*100)),"% quantile)"), legend.prop=0.2, cex.axis=2.4, cex.lab=1.9)
      abline(v = pos_year, col="black", lwd=3, lty=2)
      dev.off()
    }
    
    if (ratio > 0){
      
      sample.nonstable <- samplingfreq(tab.nonstable, tab.nonstable.seq, ratio, quantile)
      
      outputs = list('stable'=sample.stable,'nonstable'=sample.nonstable, 'tyear'=tyear)
    }
    
  } else {
    
    if ((plot == TRUE) & (dim(tab.stable)[1] > 0)) {
    layout(matrix(c(1,2,3), ncol = 3, nrow=1, byrow = TRUE))
    par(mar = c(1, 1, 1, 1), las=2, cex.main = 2.5,  mai = c(0.8, 0.6, 0.8, 0.6))
    seqdplot(tab.seq.s, with.legend = F, border = NA,main="LC distribution per year (fraction [0-1])", ylab="", legend.prop=0.2, cex.axis=1.9)
    abline(v = pos_year, col="black", lwd=3, lty=2)
    seqfplot(tab.stable.seq, idxs=1:nrow(tab.stable.seq), with.legend = F, border = NA, main="Stable LC sequences \n (All)", legend.prop=0.2, cex.axis=2.4, cex.lab=1.9)
    abline(v = pos_year, col="black", lwd=3, lty=2)
    dev.off()
    }
    
    if (ratio > 0){
      outputs = list('stable'=sample.stable, 'tyear'=tyear)
    }
    
  }
  
  if (ratio > 0){
    return(outputs)
  }
  else if (ratio == 0){
    if ((dim(tab.stable)[1] > 0) & (dim(tab.stable)[1] > 0)){
      outputs = list('tabseqstable'=tab.stable.seq, 'tabseqnonstable'=tab.nonstable.seq, 'tyear'=tyear)
    }
    else if ((dim(tab.stable)[1] == 0) & (dim(tab.nonstable)[1] > 0)){
        outputs = list('tabseqnonstable'=tab.nonstable.seq, 'tyear'=tyear)
    }
    else if ((dim(tab.stable)[1] > 0) & (dim(tab.nonstable)[1] == 0)){
        outputs = list('tabseqstable'=tab.stable.seq, 'tyear'=tyear)
    }
    return(outputs)
  }
}

# dataset
tiles <- c('tile_raisg')

lc_target_years <-c(2001,2017)
terrai_dataset <- '2004_01_01_to_2019_06_10'

lyear = c(2004:2016)
for (j in lyear){
  terrai_target_years <-c(j,j)
  
  quantile = 0.95
  
  step = 'plotting'
  
  if (step == 'plotting'){
    datasets <- c('mapbiomas')
    plot = TRUE
    ratio <- 0
  } else if (step == 'sampling'){
    #datasets <- c('mapbiomas','Copernicusnew_cf2others','MCD12Q1v6raw_LCType1')
    datasets <- c('mapbiomas')
    plot = FALSE
    #sampling
    ratio <- 0.01
    sample_dir <- paste0("E:/acocac/research/AMZ/verification/sampling/raw")
    dir.create(sample_dir, showWarnings = FALSE, recursive = T)
  }
  
  for (tile in tiles){
    
    chart_dir <- paste0("E:/acocac/research/",tile,"/post/pngs")
    dir.create(chart_dir, showWarnings = FALSE, recursive = T)
  
    data_dir <- paste0("E:/acocac/research/",tile,"/post/data")
    dir.create(data_dir, showWarnings = FALSE, recursive = T)
    
    for (d in datasets){
      
      if (d == 'mapbiomas'){
        trat <- d
      } else {
        trat <- 'prediction' #prediction mapbiomas
        ckpt <- '42497'
        epochs <- 30
      }
  
      if (d %in% c('MCD12Q1v6raw_LCType1','Copernicusnew_cf2others','MCD12Q1v6stable_LCType1')){
        schemes <-c('raw','simple') #simple
        #schemes <-c('simple') #simple
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
        
        suffix <- paste0(as.character(seq(terrai_target_years[1],terrai_target_years[2],1)),collapse="-")
        
        rlist::list.save(results, paste0(data_dir,'/','resultsraw_tyear_',suffix,'.rdata'))
        
        output <- split_seq(d, results, ratio, quantile, plot)
        
        rlist::list.save(output, paste0(data_dir,'/','resultsseq_tyear_',suffix,'.rdata'))
        
        if (ratio > 0){
          if (length(output) > 2){
            
            targetyear = output[[3]]
            
            print(paste0('Tile ',tile))
            print(paste0('Dataset ',d,' & Scheme ',scheme,' & Tyear ',targetyear))
            print(paste0('size stable = ',dim(output[[1]])[1]))
            print(paste0('size non-stable = ',dim(output[[2]])[1]))
            
            write.csv(cbind(tile=tile,output[[1]]),file=paste0(sample_dir,"/stable_",d,"_",scheme,"_",tile,"_sratio",gsub("[.]","_",ratio),"_",targetyear,".csv"), row.names=F)
            write.csv(cbind(tile=tile,output[[2]]),file=paste0(sample_dir,"/nonstable_",d,"_",scheme,"_",tile,"_sratio",gsub("[.]","_",ratio),"_",targetyear,".csv"), row.names=F)
            
          } else{
            
            targetyear = output[[2]]
            
            print(paste0('Tile ',tile))
            print(paste0('Dataset ',d,' & Scheme ',scheme,' & Tyear ',targetyear))
            print(paste0('size stable = ',dim(output[[1]])[1]))
            
            write.csv(cbind(tile=tile,output[[1]]),file=paste0(sample_dir,"/stable_",d,"_",scheme,"_",tile,"_sratio",gsub("[.]","_",ratio),"_",targetyear,".csv"), row.names=F)
          }
        }
      }
    }
  }
}