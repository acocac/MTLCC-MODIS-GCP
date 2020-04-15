require(raster)

set.seed(1)

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

gen_terrai <- function(terrai_dataset, dataset, Syear, lc_target_years, terrai_target){
  
  if (trat == 'prediction'){
    # post filest
    dir <- paste0("E:/acocac/research/",tile,"/post")
    pattern <- paste0(dataset,'_',ckpt) #MCD12Q1v6raw_LCProp1  ESAraw
  } else if (trat == 'raw') {
    # raw files
    dir <- paste0("E:/acocac/research/",tile,"/eval/pred/1_dataset/ep",epochs,"/convgru/convgru64_15_fold0_",dataset,"_",ckpt)
    pattern <- "prediction"
  } else if (trat == 'mapbiomas') {
    dir <- paste0("E:/acocac/research/",tile,"/eval/verification/mapbiomas")
    pattern <- "mapbiomas"
  }
  
  files <- list.files(path=dir, pattern="*.tif$", all.files=FALSE, full.names=TRUE,recursive=TRUE)
  #subset dataset
  files <- Filter(function(x) grepl(pattern, x), files)
  
  #subset years
  wildcard <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)),collapse="|")
  files <- Filter(function(x) grepl(wildcard, x), files)
  
  lucserie <- raster::stack(files)
  dim(lucserie[[1]])
  # plot(lucserie[[1]])
  
  watermask_file <- dir <- paste0("E:/acocac/research/",tile,"/eval/verification/watermask/2018/watermask/0_0_0.tif")
  watermask_raster <- raster(watermask_file)
  
  terrai_file <- paste0('T:/GISDATA_terra/outputs/Latin/',terrai_dataset,'/TIFF/GEOGRAPHIC/WGS84/decrease/region/classified/latin_decrease_',terrai_dataset,'.tif')
  terrai_raster <- raster(terrai_file)
  e <- extent(lucserie)
  
  terrai.crop <- crop(terrai_raster, e)
  watermask.crop <- crop(watermask_raster, e)
  
  terrai.masked = terrai.crop * watermask.crop
  
  terrai_target_clump = clumpraster(terrai.masked)
  
  terrai_final = terrai_target_clump
  
  dir.create(file.path(paste0("E:/acocac/research/",tile,"/post"), "thesis"), showWarnings = FALSE)
  
  writeRaster(terrai.crop,paste0("E:/acocac/research/",tile,"/post/thesis/terrai_raw.asc"), format="ascii", overwrite=TRUE)
  writeRaster(watermask.crop,paste0("E:/acocac/research/",tile,"/post/thesis/watermask.asc"), format="ascii", overwrite=TRUE)
  writeRaster(terrai.masked,paste0("E:/acocac/research/",tile,"/post/thesis/terrai_water.asc"), format="ascii", overwrite=TRUE)
  writeRaster(terrai_target_clump,paste0("E:/acocac/research/",tile,"/post/thesis/terrai_clumped.asc"), format="ascii", overwrite=TRUE)

  #added to compute year with maximum
  freq.tb = freq(terrai_final, useNA='no')
  freq.tb = freq.tb[which(freq.tb[,1] == terrai_target[1]):which(freq.tb[,1] == terrai_target[2]),]
  
  tyear = freq.tb[which.max(freq.tb[,2])]
  print(tyear)
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
  
  print(freq(terrai_target, useNA='no'))
  
  for (i in seq(0, 16, 1)){
    lucserie_new <- lucserie[[i+1]]
    lucserie_new <- mask(lucserie_new, terrai_target)

    writeRaster(lucserie_new,paste0(terrai_dir,"/terrai_",lc_target_years[1] + i,"_def",tyear,"_",dataset,".asc"), format="ascii", overwrite=TRUE)
  }
  
  # 
  # for (i in seq(0, 13, 1)){
  #   
  #   target.values <- function(x) { x[x<Syear_terrai] <- NA; return(x) }
  #   terrai_target <- calc(terrai.masked, target.values)
  #   if (i<11){
  #     target.values <- function(x) { x[x>Syear_terrai+i] <- NA; return(x) }
  #     terrai_target <- calc(terrai_target, target.values)
  #   } else{
  #     target.values <- function(x) { x[x>2014] <- NA; return(x) }
  #     terrai_target <- calc(terrai_target, target.values)
  #   }
  #   target.values <- function(x) { x[!is.na(x)] <- 1; return(x) }
  #   terrai_target <- calc(terrai_target, target.values)
  #   
  #   lucserie_new <- lucserie[[i+1]]
  #   lucserie_new <- mask(lucserie_new, terrai_target)
  #   
  #   writeRaster(lucserie_new,paste0(terrai_dir,"/terrai_",Syear_terrai + i,"_",dataset,".asc"), format="ascii", overwrite=TRUE)
  # }
}

# dataset
#tiles <- c('tile_0_365','tile_0_201', 'tile_0_143', 'tile_1_463', 'tile_1_438', 'tile_0_630', 'tile_1_713')
tiles <- c('tile_0_630')
#datasets <- c('merge_datasets2own', 'Copernicusraw','ESAraw', 'Copernicusnew_cf2others', 'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6raw_LCType1')
datasets <- c('mapbiomas')

for (tile in tiles){
  if ('mapbiomas'%in% datasets){
    trat <- 'mapbiomas'
  } else{
    trat <- 'prediction' #prediction mapbiomas
    ckpt <- '42497'
    tiles <- 1
    epochs <- 30
  }
  
  lc_target_years <-c(2001,2017)
  terrai_dataset <- '2004_01_01_to_2019_06_10'
  terrai_target_years <-c(2004,2010)
  
  terrai_dir <- paste0("E:/acocac/research/",tile,"/post/terrai")
  dir.create(terrai_dir, showWarnings = FALSE, recursive = T)
  
  for (d in datasets){
    gen_terrai(terrai_dataset, d, Syear_terrai, lc_target_years, terrai_target_years)
  }
}