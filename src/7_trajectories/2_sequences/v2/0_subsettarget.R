rm(list = ls())

prepare_target <- function(tyear, lc_target_years){
  
  ##analysis
  infile <- paste0(indir,'/','resultsseq_tyear_',as.character(tyear),'_simple.rdata')
  
  results_df = rlist::list.load(infile)
  
  tab = results_df[4][[1]]
  tyear = results_df[3][[1]]
  
  yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))
  
  ##determine start/end year
  start_idx <- 2 + (which(yearls == tyear)-2)
  end_idx <- 2 + (which(yearls == tyear)+2)
  
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
  
  classes <- c()
  for (year in start_idx:end_idx){
    classes <- unique(c(classes,tab[,year]))
  }
  
  target_classes <- global_classes[sort(classes)]
  target_short <- global_shortclasses[sort(classes)]
  target_colors <- global_colors[sort(classes)]
  
  alphabet <- sort(classes)
  
  ##only analyse those with DF 2yrs before/after target
  tab.lag <- seqdef(tab, start_idx:end_idx, alphabet = alphabet, states = target_short,
                    cpal = target_colors, labels = target_short, with.missing = TRUE)
  
  tabe.seq <- seqecreate(tab.lag, use.labels = FALSE)
  lc <- seqecontain(tabe.seq, event.list = c("DF"))
  tab <- tab[lc,]
  
  print(paste0('Dimensions ori data: ', dim(tab)[1]))
  
  #create stratas
  std.df <- apply(tab[,start_idx:(dim(tab)[2]-1)], 1, sd) 
  tab[,start_idx:(dim(tab)[2]-1)]
  tab.stable = tab[std.df==0,]
  tab.nonstable = tab[std.df!=0,]
  
  tab.target.data = tab.nonstable[,start_idx:(dim(tab.nonstable)[2]-1)]
  #names(tab.target.data) = paste0('t',seq(0,dim(tab.target.data)[2]-1))
  names(tab.target.data) = seq(0,dim(tab.target.data)[2]-1)
  
  tab.final = cbind('x'=tab.nonstable$x,'y'=tab.nonstable$y,tab.target.data)
  tab.final$tyear = tyear
    
  print(paste0('Dimensions target data: ', dim(tab.final)[1]))
  
  return(tab.final)
  
}

library(TraMineR, quietly = TRUE)

##settings
tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_datav2")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)

##implement cluster##
targetyears = c(2004:2016)

df_all = list()
for (j in targetyears){
  tyear <- j
  df_all[[tyear]] <-prepare_target(tyear, lc_target_years)
}

target_tb <- rbindlist(df_all, fill=TRUE)

tyear<-"2004-2016"

## save target matrix
file_name <- paste0(min(targetyears),'-',max(targetyears),'_target_tb')
file_path <- paste0(seqdata_dir,'/',file_name)
if(!file.exists(paste0(file_path,".RSav"))){
  save(target_tb, file=paste0(file_path,".RSav"))
}