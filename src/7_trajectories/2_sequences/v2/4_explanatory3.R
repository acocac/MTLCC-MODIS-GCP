rm(list = ls())

.DEBUG <- FALSE

###source
#https://github.com/amysheep/NUTrajectoryPredictionAnalysis/blob/af2170541a8959cac6aff290b60023017f93496a/TrajectoryPredictionAnalysis.ipynb

library(compareGroups)
library(caret)
library(raster)
library(doParallel)
library(dplyr)
library(mice)
library(snowfall)
library(NeuralNetTools)
library(nnet)

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
proj <- CRS('+proj=longlat +ellps=WGS84')

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

tab.target_df = as.data.frame(tab.target_geo)

Class <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("t1", "t2", "t3", "t4", "t5", "t6")
)

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

final_df = data.frame(Class, tab.target_df[,c(explanatory_int,explanatory_ext)])

###modelling
meta_params <- list(
  "xgbTree" = expand.grid(
    # XGBoost tuning parameters to optimize:
    max_depth = c(4, 8, 16, 32, 64, 128),
    eta = c(0.1, 0.2, 0.3, 0.4),
    nrounds = c(50, 100, 150, 200),
    subsample = c(0.75, 0.9),
    # XGBoost tuning parameters to hold constant:
    gamma = 0, min_child_weight = 1, colsample_bytree = 1
  ),
  "C5.0" = expand.grid(
    # C5.0
    trials = c(1, 5, 10, 25, 50, 75, 100),
    model = "tree", winnow = TRUE
  ),
  "nb" = expand.grid(
    # Naive Bayes
    fL = c(0, 0.3, 0.5, 0.8, 1, 2),
    adjust = c(1, 1.5, 2),
    usekernel = TRUE # held constant
  ),
  "multinom" = expand.grid(
    # Penalized Multinomial Regression
    decay = c(0, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 7e-1)
  ),
  "rf" = expand.grid(
    # Random Forest
    mtry = 2:5
  ),
  "nnet" = expand.grid(
    size = seq(3, 33, 3),
    decay = c(1e-4, 1e-3, 1e-2, 1e-1)
  ),
  "dnn" = expand.grid(
    layer1 = c(24, 32),
    layer2 = c(8, 16),
    layer3 = 4,
    hidden_dropout = c(0, 0.1, 0.2, 0.5),
    visible_dropout = 0
  )
)

library(caret)

cv_fit <- function(method, covars, data) {
  tuning_idx <- if (.DEBUG) 1 else 1:nrow(meta_params[[method]])
  model_control <- trainControl(
    # 5-fold cross-validation:
    method = "repeatedcv", number = ifelse(.DEBUG, 2, 5), repeats = 1,
    # Up-sample to correct for class imbalance:
    sampling = "up", summaryFunction = caret::multiClassSummary,
    # Return predicted probabilities and track progress:
    classProbs = TRUE, verboseIter = TRUE, allowParallel = TRUE
  )
  if (method %in% c("multinom", "nnet")) {
    model <- train(
      Class ~ ., data = data[, c("Class", covars)],
      trControl = model_control, na.action = na.omit,
      preProcess = c("center", "scale"),
      method = method, tuneGrid = meta_params[[method]][tuning_idx,, drop = FALSE],
      trace = FALSE # suppress nnet optimizatin info
    )
  } else if (method == "dnn") {
    model <- train(
      Class ~ ., data = data[, c("Class", covars)],
      trControl = model_control, na.action = na.omit,
      preProcess = c("center", "scale"),
      method = method, tuneGrid = meta_params[[method]][tuning_idx,, drop = FALSE],
      momentum = 0.9, learningrate = 0.05, numepochs = 150,
      learningrate_scale = 0.95, # learning rate will be mutiplied by this after every iter
      activationfun = "tanh",    # better than sigm & has a centering effect on neurons
      batchsize = 512            # batches of training data that are used to calculate error and update coefficients
    )
  } else {
    model <- train(
      Class ~ ., data = data[, c("Class", covars)],
      trControl = model_control, na.action = na.omit,
      method = method, tuneGrid = meta_params[[method]][tuning_idx,, drop = FALSE]
    )
  }
  return(model)
}

names(meta_params)

models <- c("C5.0","nb","multinom","rf")
  
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)

results<-list()
for (m in models){
  results[[m]] = cv_fit(m, explanatory_int, final_df)
}

stopCluster(cl = cl)

results_final <-results

results_final$rf

resamps <- resamples(results_final)
  
library(measures)

summary(resamps)
bwplot(resamps, metric="prAUC")



