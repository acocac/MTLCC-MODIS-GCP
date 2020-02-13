# Working directory and libraries
library(TraMineR)
library(raster)
library(tidyr)

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

split_seq <- function(dataset, results_df, ratio, quantile, plot=FALSE){

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
  
  yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  
  tab_raw = results_df[5][[1]]
  tyear = results_df[6][[1]]
  
  tab= tab_raw
  
  tab[yearls] <- lapply(tab[yearls], function(x) replace(x,x %in% 7, 6) )
  tab[yearls] <- lapply(tab[yearls], function(x) replace(x,x %in% 8, 7) )
  
  tab2 <- unite(tab, sec, yearls, sep="-")
  tab$sec = tab2$sec
  
  classes <- c()
  for (year in 3:(length(yearls)+2)){
    classes <- unique(c(classes,tab[,year]))
  }
  
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
  
  if (tyear > 2006){
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

  } else if (tyear <= 2006) {

    #order distriplot by major
    tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                      cpal = palette, labels = short_labels, with.missing = TRUE)
    
    ## Get state freq with seqmeant
    ## order of frequencies
    ord <- c(3, 4, 5, 1, 2)

    ## Sorted alphabet
    alph.s <- c("DF", "OF", "Fm", "W", "Bu")

    ## we need also to sort accordingly labels and colors
    target_classes.s <- target_classes[ord]
    target_short.s <- target_short[ord]
    target_colors.s <- target_colors[ord]

    ## Define sequence object with sorted states
    tab.seq.s <- seqdef(tab.seq, alphabet = alph.s, states = target_short.s,
                        labels = target_classes.s, cpal = target_colors.s,with.missing = TRUE)


  } 
  
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
      png(filename=paste0(chart_dir,"/seqoutputs_",dataset,"_def",tyear,"_",scheme,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
    } else{
      png(filename=paste0(chart_dir,"/seqoutputs_",dataset,"_def",tyear,"_",scheme,".png"), width = sizew, height = sizeh, units = "px",  antialias = "cleartype")
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
      outputs = list('tabseqstable'=tab.stable.seq, 'tabseqnonstable'=tab.nonstable.seq, 'tyear'=tyear, 'tab'=tab)
    }
    else if ((dim(tab.stable)[1] == 0) & (dim(tab.nonstable)[1] > 0)){
      outputs = list('tabseqnonstable'=tab.nonstable.seq, 'tyear'=tyear, 'tab'=tab)
    }
    else if ((dim(tab.stable)[1] > 0) & (dim(tab.nonstable)[1] == 0)){
      outputs = list('tabseqstable'=tab.stable.seq, 'tyear'=tyear, 'tab'=tab)
    }
    return(outputs)
  }
}

tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectoriesep/charts")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)

##analysis with simple classes
targetyears = c(2004:2006)
lc_target_years <-c(2001,2019)

for (j in targetyears){
  terrai_target <- j
  print(terrai_target)
  
  quantile = 0.95
  
  step = 'plotting'
  
  if (step == 'plotting'){
    datasets <- c('mtlcc')
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
  
  chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/pngs")
  dir.create(chart_dir, showWarnings = FALSE, recursive = T)
    
  data_dir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
  dir.create(data_dir, showWarnings = FALSE, recursive = T)
    
  for (d in datasets){
      
      trat <- d
      scheme <- 'simple' 
      
      infile <- paste0(indir,'/','resultsraw_tyear_',as.character(terrai_target),'_raw.rdata')
      
      if(file.exists(infile)){ 
          results_df = rlist::list.load(infile)
          
          output <- split_seq(d, results_df, ratio, quantile, plot)
          
          rlist::list.save(output, paste0(data_dir,'/','resultsseq_tyear_',terrai_target,'_',scheme,'.rdata'))
    }
  }
}