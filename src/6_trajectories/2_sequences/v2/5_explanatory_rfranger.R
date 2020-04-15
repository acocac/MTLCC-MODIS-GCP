rm(list = ls())

## Cross-validation of soil properties adapted from https://github.com/ISRICWorldSoil/SoilGrids250m/blob/master/grids/cv/cv_functions.R
## Tom.hengl@gmail.com and Amanda Ramcharan <a.m.ramcharan@gmail.com>

list.of.packages <- c("nnet", "plyr", "ROCR", "randomForest", "plyr", "parallel", "psych", "mda", "h2o", "dismo", "grDevices", "snowfall", "hexbin", "lattice", "ranger", "xgboost", "doParallel", "caret")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


## --------------------------------------------------------------
## Build cross validation function for soil properties:
## --------------------------------------------------------------

## predict soil properties in parallel:
predict_parallelP <- function(j, sel, varn, formulaString, rmatrix, idcol, method, cpus, Nsub=1e4, remove_duplicates=FALSE, outdir){
  s.train <- rmatrix[!sel==j,]
  if(remove_duplicates==TRUE){
    ## TH: optional - check how does model performs without the knowledge of the 3D dimension
    sel.dup = !duplicated(s.train[,idcol])
    s.train <- s.train[sel.dup,]
  }
  s.test <- rmatrix[sel==j,]
  n.l <- dim(s.test)[1]
  if(missing(Nsub)){ Nsub = length(all.vars(formulaString))*50 }
  if(Nsub>nrow(s.train)){ Nsub = nrow(s.train) }
  if(method=="h2o"){
    ## select only complete point pairs
    train.hex <- as.h2o(s.train[complete.cases(s.train[,all.vars(formulaString)]),all.vars(formulaString)], destination_frame = "train.hex")
    gm1 <- h2o.randomForest(y=1, x=2:length(all.vars(formulaString)), training_frame=train.hex) 
    gm2 <- h2o.deeplearning(y=1, x=2:length(all.vars(formulaString)), training_frame=train.hex)
    test.hex <- as.h2o(s.test[,all.vars(formulaString)], destination_frame = "test.hex")
    v1 <- as.data.frame(h2o.predict(gm1, test.hex, na.action=na.pass))$predict
    gm1.w = gm1@model$training_metrics@metrics$r2
    v2 <- as.data.frame(h2o.predict(gm2, test.hex, na.action=na.pass))$predict
    gm2.w = gm2@model$training_metrics@metrics$r2
    ## mean prediction based on accuracy:
    pred <- rowSums(cbind(v1*gm1.w, v2*gm2.w))/(gm1.w+gm2.w)
    gc()
    h2o.removeAll()
  }
  if(method=="caret"){
    test = s.test[,all.vars(formulaString)]
    ## tuning parameters:
    cl <- makeCluster(cpus)
    registerDoParallel(cl)
    ctrl <- trainControl(method="repeatedcv", number=3, repeats=1)
    #gb.tuneGrid <- expand.grid(eta = c(0.3,0.4), nrounds = c(50,100), max_depth = 2:3, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1)
    rf.tuneGrid <- expand.grid(mtry = seq(2,6,by=1))
    #rf.tuneGrid <- expand.grid(mtry = seq(4,8,by=2))
    ## fine-tune RF parameters:
    t.mrfX <- caret::train(formulaString, data=s.train[sample.int(nrow(s.train), Nsub),], method="rf", trControl=ctrl, tuneGrid=rf.tuneGrid)

    gm1 <- ranger(formulaString, data=s.train, write.forest=TRUE, mtry=t.mrfX$bestTune$mtry, importance="permutation")
    #gm1.w = 1/gm1$prediction.error
    #gm2 <- caret::train(formulaString, data=s.train, method="xgbTree", trControl=ctrl, tuneGrid=gb.tuneGrid)
    #gm2.w = 1/(min(gm2$results$RMSE, na.rm=TRUE)^2)
    #v1 <- predict(gm1, test, na.action=na.pass)$predictions
    #v2 <- predict(gm2, test, na.action=na.pass)
    #pred <- rowSums(cbind(v1*gm1.w, v2*gm2.w))/(gm1.w+gm2.w)
    
    save(gm1, file=paste0(outdir,"/","cRF_model.RData"))
    pred <- predict(gm1, test, na.action=na.pass)$predictions
    
  }
  if(method=="ranger"){
    gm <- ranger(formulaString, data=s.train, write.forest=TRUE, num.trees=85, importance="permutation")
    #saveRDS(gm, file=paste0(outdir,"/","ranger_mRF_", varn,".RData"), compress=T, version=3.5.0)
    #save(gm, file=paste0(outdir,"/","ranger3_mRF_", varn,"_model.rdata"))
    save(gm, file=paste0(outdir,"/","mRF_model.RData"))
    xImp = as.list(ranger::importance(gm))
    
    save(xImp, file=paste0(outdir,"/","mRF_imp.RSav"))
    
    pred <- predict(gm, s.test, na.action = na.pass)$predictions 
  }
  if(method=="party"){
    gm <- cforest(formulaString, data=s.train, control = cforest_unbiased(mtry = 2, ntree = 85))
    #saveRDS(gm, file=paste0(outdir,"/","ranger_mRF_", varn,".RData"), compress=T, version=3.5.0)
    save(gm, file=paste0(outdir,"/","party_mRF_", varn,"_model.RData"))

    pred <- party::predict(gm, s.test, type="response")
    print(pred)
  }
  obs.pred <- as.data.frame(list(s.test[,varn], pred))
  names(obs.pred) = c("Observed", "Predicted")
  obs.pred[,idcol] <- s.test[,idcol]
  obs.pred$fold = j
  return(obs.pred)
}

cv_numeric <- function(formulaString, rmatrix, nfold, idcol, cpus, method="ranger", Log=FALSE, LLO=TRUE, outdir){     
  varn = all.vars(formulaString)[1]
  message(paste("Running ", nfold, "-fold cross validation with model re-fitting method ", method," ...", sep=""))
  if(nfold > nrow(rmatrix)){ 
    stop("'nfold' argument must not exceed total number of points") 
  }
  
  sel <- dismo::kfold(rmatrix, k=nfold, by=rmatrix$group)
  message(paste0("Subsetting observations by '", idcol, "'"))
  
  if(missing(cpus)){ 
    if(method=="randomForest"){
      cpus = nfold
    } else { 
      cpus <- parallel::detectCores(all.tests = FALSE, logical = FALSE) 
    }
  }
  if(method=="h2o"){
    out <- list()
    for(j in 1:nfold){ 
      out[[j]] <- predict_parallelP(j, sel=sel, varn=varn, formulaString=formulaString, rmatrix=rmatrix, idcol=idcol, method=method, cpus=1)
    }
  }
  if(method=="caret"){
    out <- list()
    for(j in 1:nfold){ 
      out[[j]] <- predict_parallelP(j, sel=sel, varn=varn, formulaString=formulaString, rmatrix=rmatrix, idcol=idcol, method=method, cpus=cpus, outdir=outdir)
    }
  }
  if(method=="ranger"){
    snowfall::sfInit(parallel=TRUE, cpus=ifelse(nfold>cpus, cpus, nfold))
    snowfall::sfExport("predict_parallelP","idcol","formulaString","rmatrix","sel","varn","method","outdir")
    snowfall::sfLibrary(package="plyr", character.only=TRUE)
    snowfall::sfLibrary(package="ranger", character.only=TRUE)
    out <- snowfall::sfLapply(1:nfold, function(j){predict_parallelP(j, sel=sel, varn=varn, formulaString=formulaString, rmatrix=rmatrix, idcol=idcol, method=method, outdir=outdir)})
    snowfall::sfStop()
  }
  if(method=="party"){
    snowfall::sfInit(parallel=TRUE, cpus=ifelse(nfold>cpus, cpus, nfold))
    snowfall::sfExport("predict_parallelP","idcol","formulaString","rmatrix","sel","varn","method","outdir")
    snowfall::sfLibrary(package="plyr", character.only=TRUE)
    snowfall::sfLibrary(package="party", character.only=TRUE)
    out <- snowfall::sfLapply(1:nfold, function(j){predict_parallelP(j, sel=sel, varn=varn, formulaString=formulaString, rmatrix=rmatrix, idcol=idcol, method=method, outdir=outdir)})
    snowfall::sfStop()
  }
  ## calculate mean accuracy:
  out <- plyr::rbind.fill(out)
  OA =confusionMatrix(out$Observed, out$Predicted)$overall["Accuracy"]
  
  # 
  # ME = mean(out$Observed - out$Predicted, na.rm=TRUE) 
  # MAE = mean(abs(out$Observed - out$Predicted), na.rm=TRUE)
  # RMSE = sqrt(mean((out$Observed - out$Predicted)^2, na.rm=TRUE))
  # ## https://en.wikipedia.org/wiki/Coefficient_of_determination
  # #R.squared = 1-sum((out$Observed - out$Predicted)^2, na.rm=TRUE)/(var(out$Observed, na.rm=TRUE)*sum(!is.na(out$Observed)))
  # R.squared = 1-var(out$Observed - out$Predicted, na.rm=TRUE)/var(out$Observed, na.rm=TRUE)
  if(Log==TRUE){
    ## If the variable is log-normal then logR.squared is probably more correct
    logRMSE = sqrt(mean((log1p(out$Observed) - log1p(out$Predicted))^2, na.rm=TRUE))
    #logR.squared = 1-sum((log1p(out$Observed) - log1p(out$Predicted))^2, na.rm=TRUE)/(var(log1p(out$Observed), na.rm=TRUE)*sum(!is.na(out$Observed)))
    logR.squared = 1-var(log1p(out$Observed) - log1p(out$Predicted), na.rm=TRUE)/var(log1p(out$Observed), na.rm=TRUE)
    cv.r <- list(out, data.frame(ME=ME, MAE=MAE, RMSE=RMSE, R.squared=R.squared, logRMSE=logRMSE, logR.squared=logR.squared))
  } else {
    cv.r <- list(out, data.frame(OA=OA))
  }
  names(cv.r) <- c("CV_residuals", "Summary")
  return(cv.r)
  closeAllConnections()
}

################################################
################################################

library(sp)
library(randomForest)
library(nnet)
library(plotKML)
library(GSIF)
library(plyr)
library(ROCR)
library(snowfall)
library(mda)
library(psych)
library(hexbin)
library(gridExtra)
library(lattice)
library(grDevices)
library(h2o)
library(scales)
library(ranger)
library(xgboost)
library(caret)
library(doParallel)
library(RCurl)
library(viridis)
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(grid)
library(party)
library(dplyr)

tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatoryv2")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clustersv2")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distancesv2")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_datav2")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)

##aux
aux_dir <- "F:/acoca/research/gee/dataset/AMZ/implementation"

###start process
fn <- 'WARD_clustersk6_2004-2016_minperiod12_TRATE_OM_target.RSav'

file_name <- paste0('auxdata_',fn)
file_path <- paste0(explanatory_dir,'/',file_name)

if(file.exists(paste0(file_path))){
  load(file_path)
} else{
  load(paste0(clusters_dir,'/',fn))
  
  access <- raster(paste0(aux_dir,'/ancillary/processed/external/access.tif'))
  dem <- raster(paste0(aux_dir,'/ancillary/processed/external/srtm.tif'))
  slope <- raster(paste0(aux_dir,'/ancillary/processed/external/slope.tif'))
  precipitation <- raster(paste0(aux_dir,'/ancillary/processed/external/bio12.tif'))
  pas_conservation <- raster(paste0(aux_dir,'/ancillary/processed/external/distance_PAs_conservation_AMZ.tif'))
  pas_exploitation <- raster(paste0(aux_dir,'/ancillary/processed/external/distance_PAs_exploitation_AMZ.tif'))
  
  c1 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c1.tif'))
  c2 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c2.tif'))
  c3 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c3.tif'))
  c4 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c4.tif'))
  c5 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c5.tif'))
  c6 <- raster(paste0(aux_dir,'/ancillary/processed/internal/distance_c6.tif'))
  
  ##extract raster values
  ##add and prepare aux columns
  tab.target_geo = unique_seq_subset
  coordinates(tab.target_geo) <- c("x", "y")
  mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))
  
  #external
  dem_val =raster::extract(dem, mypoints)
  slope_val =raster::extract(slope, mypoints)
  access_val =raster::extract(access, mypoints)
  prec_val =raster::extract(precipitation, mypoints)
  pascon_val =raster::extract(pas_conservation, mypoints)
  pasexp_val =raster::extract(pas_exploitation, mypoints)
  
  #internal
  c1_val =raster::extract(c1, mypoints)
  c2_val =raster::extract(c2, mypoints)
  c3_val =raster::extract(c3, mypoints)
  c4_val =raster::extract(c4, mypoints)
  c5_val =raster::extract(c5, mypoints)
  c6_val =raster::extract(c6, mypoints)
  
  #external values
  tab.target_geo$access = (access_val)/(60*24)
  tab.target_geo$dem = dem_val
  tab.target_geo$slope = slope_val
  tab.target_geo$prec = prec_val
  tab.target_geo$pascon = (pascon_val * 231.91560544825498) / 1000
  tab.target_geo$pasexp = (pasexp_val * 231.91560544825498) / 1000
  
  #internal values
  tab.target_geo$c1 = (c1_val * 231.91560544825498) / 1000
  tab.target_geo$c2 = (c2_val * 231.91560544825498) / 1000
  tab.target_geo$c3 = (c3_val * 231.91560544825498) / 1000
  tab.target_geo$c4 = (c4_val * 231.91560544825498) / 1000
  tab.target_geo$c5 = (c5_val * 231.91560544825498) / 1000
  tab.target_geo$c6 = (c6_val * 231.91560544825498) / 1000
  
  
  save(tab.target_geo, file=paste0(file_path))
}

startP = 'load'

if (start = 'raw'){
  tab.target_df = as.data.frame(tab.target_geo)
  
  group <- factor(
    tab.target_geo$clusters,
    c(1, 2, 3, 4, 5, 6),
    c("t1", "t2", "t3", "t4", "t5", "t6")
  )
  
  final_df = data.frame(group, tab.target_df[,17:28])
  
}

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

# This code can be repeated for all soil properties - 2 soil properties provided here
cov <- c('access','dem','slope','prec','pascon','pasexp')
cov <-explanatory_int


formulaStringClay = as.formula(paste('group ~', paste(cov, collapse="+")))
df.target <- final_df[,all.vars(formulaStringClay)]

#### Run cross validation ####
results_cv <- cv_numeric(formulaStringClay, df.target, nfold=10, idcol="group",outdir=explanatory_dir)

file_name <- paste0("mRF_results_int.RData")
file_path <- paste0(explanatory_dir,'/',file_name)

save(results_cv, file=paste0(file_path))

model_fn = paste0(explanatory_dir,"/","mRF_model_int.RData")
load(model_fn) # all exist

importance_fn = paste0(explanatory_dir,"/","mRF_imp_both.RSav")
load(importance_fn) # all exist

xl = as.list(ranger::importance(gm))
xl = t(data.frame(xl[order(unlist(xl), decreasing=TRUE)[1:6]]))

# gm
