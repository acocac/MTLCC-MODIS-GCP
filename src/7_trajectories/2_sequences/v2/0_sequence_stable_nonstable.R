rm(list = ls())

# Working directory and libraries
library(TraMineR)
library(raster)
library(tidyr)
library(future.apply)
library(data.table)
library(ggplot2)
library(viridis)
library(labelled)
library(tidyr)
library(hrbrthemes)
library(tidyr)
require(dplyr)

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
                    '#db00ff', 
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
  
  sum_df = data.frame("dyear"=tyear,"stable"=dim(tab.stable)[1], "nonstable"=dim(tab.nonstable)[1])
  
  return(sum_df)
  
}

stable_nontable <- function(j){
  terrai_target <- j
  
  quantile = 0.95
  
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
  
  for (d in datasets){
    
    trat <- d
    scheme <- 'simple' 
    
    infile <- paste0(indir,'/','resultsraw_tyear_',as.character(terrai_target),'_raw.rdata')
    
    if(file.exists(infile)){ 
      results_df = rlist::list.load(infile)
      output <- split_seq(d, results_df, ratio, quantile, plot)
      
      return(output)
    }
  }
}

tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
charts_dir <- paste0("E:/acocac/research/",tile,"/trajectories/chartsv2")
dir.create(charts_dir, showWarnings = FALSE, recursive = T)

##analysis with simple classes
plan(multiprocess, workers = 13) ## Parallelize using four cores
set.seed(123)

targetyears = c(2001:2018)
lc_target_years <-c(2001,2019)
step = 'plotting'

stability = future.apply::future_lapply(targetyears, FUN = stable_nontable, future.seed = TRUE)

stability_tb <- rbindlist(stability, fill=TRUE)

target_reshape <- melt(stability_tb, id = c("dyear"), value.name = "pixels")

n <- target_reshape[dyear %in% (2004:2018), .(n = sum(pixels)), by = dyear]$n
etiquettes <- paste0(2004:2018, "\n(", round(n*10^-3,0), "K)")

png(file = paste0(charts_dir,"/stablenonstable_2004-2018.png"), width = 850, height = 450, res = 90)
target_reshape %>%
  group_by(dyear) %>%
  mutate(percent = pixels / sum(pixels)) %>%
  ggplot(aes(x=dyear, y=percent, fill=variable)) +
  geom_bar(position="stack", stat="identity") +
  scale_size(name="", range=c(0,10))  +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5)) +
  theme_ipsum_rc(axis_title_size = 16, base_size=10) +
  theme(legend.position="bottom") +
  scale_x_continuous(breaks = 2004:2018, labels = etiquettes) +
  scale_y_continuous(labels = scales::percent) +
  # labs(x="Deforested year", y="Proportion",
  #      title="2001-2019 stable vs non-stable LC sequences",
  #      subtitle="Organised by deforested year incl. total sequences in thousand (K)") +
  labs(x="Deforested year", y="Proportion") +
  theme(legend.text=element_text(size=14)) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank()) +
  scale_fill_manual("",values=c("gray","black"),breaks=c('stable','nonstable'),
                    labels=c('Stable','Non-Stable'))

dev.off()