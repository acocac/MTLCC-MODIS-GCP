rm(list = ls())

###source
#https://github.com/amysheep/NUTrajectoryPredictionAnalysis/blob/af2170541a8959cac6aff290b60023017f93496a/TrajectoryPredictionAnalysis.ipynb

library(compareGroups)
library(caret)
library(raster)
library(doParallel)
library(dplyr)
library(snowfall)
library(labelled)

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
fn <- 'WARD_clustersk6_2004-2016_minperiod12_TRATE_OM_unique.RSav'

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

group <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("t1", "t2", "t3", "t4", "t5", "t6")
)

final_df = data.frame(group, tab.target_df[,17:28])

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

#final_df = data.frame(group, tab.target_df[,c(explanatory_int,explanatory_ext)])
final_df = data.frame(group, tab.target_df[,explanatory_ext])

preProcValues <- preProcess(final_df[-1], method = c("center", "scale", "BoxCox"))

trainTransformed <- predict(preProcValues, final_df[-1])

m=ggplot(trainTransformed,aes(x=slope,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=slope,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 5,
                     classProbs = TRUE,
                     savePredictions = TRUE)


no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)     

set.seed(476)
glmnetGrid <- expand.grid(alpha = c(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1),
                          lambda = seq(0.0, 0.04, length = 20))
glmnetTune <- train(x = final_df[, -1],
                    y = final_df$group,
                    method = "glmnet",
                    tuneGrid = glmnetGrid,
                    preProc = c("center", "scale"),
                    metric = "Kappa",
                    family = "multinomial",
                    trControl = ctrl)

impDF <- varImp(glmnetTune$finalModel, scale=F)
coef(glmnetTune$finalModel, s=0)

stopCluster(cl = cl)

summary(glmnetTune$finalModel)

#summary table - opt1
t=compareGroups(group~.,data=final_df,include.miss=F,max.ylev =6)
t1=createTable(t,show.all = T,digits=2)

#summary table - opt1
dependent <- "group"
tab <- summary_factorlist(
  to_factor(final_df), dependent, explanatory_ext,
  p=TRUE, column = TRUE, total_col = TRUE
)
knitr::kable(tab, row.names = FALSE)

#multinomial
final_df$group2 <- relevel(final_df$group, 2)
regm <- multinom(
  group2 ~ access + dem + slope + prec + pascon + pasexp,
  data = to_factor(final_df)
)

vif(final_df[-1])

tmp <- JLutils::tidy_detailed(regm, exponentiate = T, conf.int = TRUE)
tmp <- tmp[tmp$term != "(Intercept)", ]
ggplot(tmp) +
  aes(x = label, y = estimate, ymin = conf.low, ymax = conf.high, color = y.level) +
  geom_hline(yintercept = 1, color = "gray25", linetype = "dotted") +
  geom_errorbar(position = position_dodge(0.5), width = 0) +
  geom_point(position = position_dodge(width = 0.5)) +
  scale_y_log10() +
  coord_flip() +
  xlab("Factors") + ylab("Odds Ratios") +
  theme_light() +
  theme(panel.grid.major.y = element_blank())

sqrt(vif(regm)) > 2 # problem?

vif(regm)

D_Var = "group2"
Dataset_1 <- cbind(final_df,D_Var_1 = as.numeric(final_df[,D_Var]))
D_Var_1 <- "D_Var_1"

varlist <- colnames(final_df)[colnames(final_df) %in% explanatory_ext]
Formula <- formula(paste(paste(D_Var_1,"~ "), paste(varlist, collapse=" + ")))
fit <- lm(Formula, data=Dataset_1)
n <- rownames(alias(fit)$Complete)
tempNames <- varlist[!varlist %in% n]

library(Boruta)
library(gtools)

####   Variable Reduction Using Baruta
Boruta(formula(paste0(D_Var,"~.")),data=final_df[,c(D_Var,tempNames)],doTrace=2)->Bor.son;
stats<-subset(attStats(Bor.son),decision == "Confirmed");
tempNames <- rownames(stats, do.NULL = TRUE, prefix = "row")

####   Variable Reduction Using VIF
if (length(tempNames) >1) {
  for (i in c(1:length(tempNames))){        
    Formula <- formula(paste(paste(D_Var_1,"~ "), paste(tempNames, collapse=" + ")))
    fit <- lm(Formula, data=Dataset_1)
    VIF_Data <- as.data.frame(vif(fit))
    VIF_Data$Vars <- rownames(VIF_Data, do.NULL = TRUE, prefix = "row")
    VIF_Data <- data.frame(VIF_Data[order(-VIF_Data[,1]), ],row.names=NULL)
    if(VIF_Data[1,1] <= 3)break
    tempNames = VIF_Data[-1,2]
    if(length(tempNames) == 1)break
  }
}

##### Defining All (2^n-1) Possible Combinations of Selected Variables
for (m in c(1:length(tempNames))){
  y <- combinations(length(tempNames),m,tempNames,repeats=FALSE)
  
  for (n in c(1:length(y[,1]))){
    y1 <- c(y[n,])
    Var_List <- paste(y1,collapse=" + ")
    
    #Split Data into Development and Validation
    # Data partition
    inTraining <- createDataPartition(final_df$group, p = .50, list = FALSE, times = 1)
    
    Dataset_D  <- final_df[inTraining,]
    Dataset_V   <- final_df[-inTraining,]
    
    CalculateConcordance <- function (myMod){       
      fitted <- data.frame (cbind (Dataset_D[,D_Var], myMod$fitted.values)) # actuals and fitted       
      colnames(fitted) <- c('response','score') # rename columns   
      ones <- fitted[fitted$response==1, ] # Subset ones        
      zeros <- fitted[fitted$response==0, ] # Subsetzeros        
      totalPairs <- nrow (ones) * nrow (zeros) # calculate total number of pairs to check        
      conc <- sum (c (vapply (ones$score, function(x) {((x > zeros$score))}, FUN.VALUE=logical(nrow(zeros)))))        
      disc <- totalPairs - conc
      # Calc concordance, discordance and ties
      concordance <- conc/totalPairs        
      discordance <- disc/totalPairs        
      tiesPercent <- (1-concordance-discordance)        
      return(list("Concordance"=concordance, "Discordance"=discordance,"Tied"=tiesPercent, "Pairs"=totalPairs))        
    }
  }
}
## parallel process ##
#cluster
# Calculate the number of cores
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)                #register the cluster

# fitting multinomial logistic regression -multinom function from nnet package
set.seed(06132017)
model1<- train(group~.,
              data=final_df, trControl=train_control, method="multinom",
              na.action=na.omit,savePredictions = TRUE, trace=F, allowParallel = TRUE)

stopCluster(cl = cl)

# optimized model
finalmodel1 <- summary(model1$finalModel, Wald=T)

##check p-values
z <- finalmodel1$coefficients/finalmodel1$standard.errors
# 2-tailed Wald z tests to test significance of coefficients
p <- (1 - pnorm(abs(z), 0, 1)) * 2

##model 2
final_df$re_group=relevel(final_df$group,ref="2")

cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)                #register the cluster

# fitting multinomial logistic regression -multinom function from nnet package
set.seed(06132017)
model2<- train(re_group~access + dem + slope + prec + pascon + pasexp,
              data=final_df,  method="multinom",
              na.action=na.omit,savePredictions = TRUE,
              trace=F, allowParallel = TRUE)

stopCluster(cl = cl)

finalmodel2 <- summary(model2$finalModel, Wald=T)

##model 3
final_df$re_group=relevel(final_df$group,ref="2")

cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)                #register the cluster

#Step 3: Complete case multinomial logistic regression (n=417)
train_control<- trainControl(method="cv", number=10,savePredictions = TRUE,sampling="up")

# fitting multinomial logistic regression -multinom function from nnet package
set.seed(06132017)
model3<- train(re_group~access + dem + slope + pascon + pasexp,
               data=final_df, trControl=train_control, method="multinom",
               na.action=na.omit,savePredictions = TRUE,
               tuneGrid=expand.grid(.decay=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),trace=F, allowParallel = TRUE)

stopCluster(cl = cl)

finalmodel3 <- summary(model3$finalModel, Wald=T)

library(broom)

tidy(model1$finalModel)->m1
tidy(model2$finalModel)->m2
tidy(model3$finalModel)->m3

table_func <- function(m){
  #or <- round(exp(m$estimate),3)
  #or_low <- round(exp(m$estimate-1.96*m$std.error),3)
  #or_high <- round(exp(m$estimate+1.96*m$std.error),3)
  group <- m$y.level
  var <- m$term
  p <- round(m$p.value,3)
  #out <- data.frame(cbind(group,var,or,or_low,or_high,p))
  out <- data.frame(cbind(group,var,p))

  return(out)
}

table_func(m1)
table_func(m2)
table_func(m3)

filter(model1a$pred,decay==0.4)%>%select(obs,pred)->cm
confusionMatrix(data=cm$pred,reference=cm$obs)

#Step 4: Model with multiple imputation (5 different sets of imputations) for missing covariates (R package MICE-multiple imputation by chained equations)

mdpt <- md.pattern(select(final_df,access,dem,slope,pascon,pasexp))


bar.theme_nolegend = theme_classic(base_size = 12, base_family = "Arial") + theme(plot.title = element_text(face="bold", colour="#000000", size=14, vjust=1.3), axis.title.x = element_text(face="plain", colour="#000000", size=15, vjust=-0.3),
                                                                                  axis.text.x  = element_text(angle=90, vjust=0.5, size=9), axis.title.y = element_text(face="plain", colour="#000000", size=15, vjust=0.8), axis.text.y  = element_text(angle=0, vjust=0.5, size=12)) +
  theme(plot.background = element_blank(),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),panel.border = element_blank()) +
  theme(axis.line = element_line(color = 'black')) + theme(legend.position = "none")

model3<- train(re_group~access + dem + slope + pascon + pasexp,
               data=final_df, trControl=train_control, method="multinom",
               na.action=na.omit,savePredictions = TRUE,
               tuneGrid=expand.grid(.decay=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),trace=F, allowParallel = TRUE)



###sensitibity

final_df = data.frame(group, tab.target_df[,17:28])

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c1','c2','c3','c4','c5','c6')

final_df = data.frame(group, tab.target_df[,c(explanatory_int,explanatory_ext)])
#final_df = data.frame(group, tab.target_df[,explanatory_ext])
names(final_df)

descrCor <- cor(final_df[-1])
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)

# Calculate the number of cores
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)                #register the cluster

maxit.nnet = 5000
rang.nnet = 0.7
MaxNWts.nnet = 2000

#Step 3: Complete case multinomial logistic regression (n=417)
train_control<- trainControl(method="cv", number=10,savePredictions = TRUE,sampling="up")

# fitting multinomial logistic regression -multinom function from nnet package
set.seed(06132017)
model1<- train(group~.,
               data=final_df, trControl=train_control, method="multinom",
               na.action=na.omit,savePredictions = TRUE,
               tuneGrid=expand.grid(.decay=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),trace=F, allowParallel = TRUE)

stopCluster(cl = cl)


#caret
final_df = data.frame(group, tab.target_df[,17:28])

explanatory_ext <- c('access','dem','slope','prec','pascon','pasexp')
explanatory_int <- c('c2','c3','c4','c5','c6')

final_df = data.frame(group, tab.target_df[,c(explanatory_int,explanatory_ext)])
#final_df = data.frame(group, tab.target_df[,explanatory_int])
names(final_df)


# =============================================================================================
# SPLIT TRAINING AND VALIDATION DATA
# =============================================================================================
final_df <- final_df[complete.cases(final_df),]
target_df <- final_df

set.seed(1000)
# Data partition
inTraining <- createDataPartition(target_df$group, p = .80, list = FALSE, times = 1)

training <- target_df[inTraining,]
testing  <- target_df[-inTraining,]

xtraining <- training[,2:dim(final_df)[2]]
ytraining <- training$group

xtesting <- testing[,2:dim(final_df)[2]]
ytesting <- testing$group

# =============================================================================================
# CROSS VALIDATION WITH DIFFERENTS CLASSIFIERS
# =============================================================================================
set.seed(40)

control <- trainControl(method = "LGOCV", number = 3, verboseIter = T)


no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster
registerDoParallel(cl)

set.seed(40)

# put your column to classify
rf <- caret::train(x = xtraining,
                   y = as.factor(ytraining),
                   method = "rf",
                   ntree = 1000,
                   trControl = control)
# 
# # put your column to classify
# svm <- caret::train(x = xtraining,
#                     y = as.factor(ytraining),
#                     method = "svmRadial",
#                     trControl = control)
# 
# # put your column to classify
# mlp <- caret::train(x = xtraining,
#                     y = as.factor(ytraining),
#                     method = "mlp",
#                     trControl = control)
# 
# # put your column to classify
# nb <- caret::train(x = xtraining,
#                    y = as.factor(ytraining),
#                    method = "nb",
#                    trControl = control)
# 
# # put your column to classify
# cart <- caret::train(x = xtraining,
#                      y = as.factor(ytraining),
#                      method = "rpart",
#                      trControl = control)
# 
# mlog <- caret::train(x = xtraining,
#                      y = as.factor(ytraining), trControl=control, method="multinom",
#                     na.action=na.omit,savePredictions = TRUE,
#                     tuneGrid=expand.grid(.decay=c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),trace=F)
# 
# 
# ann_center <- train(x = xtraining,
#                    y = as.factor(ytraining),
#                    trControl = control,
#                    method = "nnet",
#                    maxit = maxit.nnet, 
#                    rang = rang.nnet,
#                    MaxNWts = MaxNWts.nnet,
#                    preProcess = c("center", "scale"),
#                    trace = F, 
#                    linout = F)

stopCluster(cl = cl)

### Section 4: ANNs models
## Section 4a: ANNs model settings
#ANN parameters
decay.tune = c(0.0001, 0.001, 0.01, 0.1)
#decay.tune = c(0.01, 0.1)
size = seq(1, 50,by=2)
maxit.nnet = 5000
rang.nnet = 0.7
MaxNWts.nnet = 2000

#tuning grid for train caret function
my.grid <- expand.grid(.decay = decay.tune, .size = size)

n.repeats = 3
n.resampling = 3

#create a list of seed, here change the seed for each resampling
set.seed(40)
length.seeds = (n.repeats*n.resampling)+1
n.tune.parameters = length(decay.tune)*length(size)
seeds <- vector(mode = "list", length = length.seeds)#length is = (n_repeats*nresampling)+1
for(i in 1:length.seeds) seeds[[i]]<- sample.int(n=1000, n.tune.parameters) #(n.tune.parameters = number of tuning parameters)
seeds[[length.seeds]]<-sample.int(1000, 1)#for the last model

#create a control object for the models, implementing 10-crossvalidation repeated 10 times
fit.Control <- trainControl(
  method = "repeatedcv",
  number = n.resampling, ## k-fold CV
  repeats = n.repeats, ## iterations
  classProbs=TRUE,
  savePred = TRUE,
  seeds = seeds
)

## Section 4b: Run ANNs models 
## parallel process ##
#cluster
# Calculate the number of cores
no_cores <- detectCores() - 1
cl <- makeCluster(no_cores, type = "SOCK")    #create a cluster 
registerDoParallel(cl)                #register the cluster 

set.seed(40)
## foreach or lapply would do this faster
fit.model <- train(x = xtraining,
                   y = as.factor(ytraining),
                   trControl = fit.Control,
                   method = "nnet",
                   maxit = maxit.nnet, 
                   rang = rang.nnet,
                   MaxNWts = MaxNWts.nnet,
                   preProcess = c("center", "scale"),
                   tuneGrid = my.grid, 
                   trace = F, 
                   linout = F
)

stopCluster(cl = cl)

# =============================================================================================
# RESULTS: CROSS - VALIDATION WITH DIFFERENTS CLASSIFIERS
# =============================================================================================
resamps <- resamples(list(ANN_ft2 = fit.model,
                          ANN_ft = fit.model
))

resamps
summary(resamps)
bwplot(resamps)

varImp(rf$finalModel)

variableImportance=data.frame(varImp(rf$finalModel))
variableImportance$Overall = apply(variableImportance, 1, median) # calculate importance over all classes
variableImportance$Variables=row.names(variableImportance)
variableImportance = variableImportance[,c('Overall', 'Variables')]
names(variableImportance)=c('Importance', 'Variables')
variableImportance=variableImportance[order(variableImportance$Importance),]
variableImportance$id = 1:nrow(variableImportance)


varImpPlot=ggplot(variableImportance, aes(as.numeric(id), as.numeric(Importance), label=Variables)) 
varImpPlot=varImpPlot+geom_point(size = 3, shape=21, fill="#000000", colour="#000000")
varImpPlot=varImpPlot+geom_line(size = 0.5, colour="#000000", linetype=1)
varImpPlot=varImpPlot+geom_text(size=3, vjust=-1.5, check_overlap = TRUE)
varImpPlot=varImpPlot+coord_cartesian()
varImpPlot=varImpPlot+ggtitle("Feature Importance")
varImpPlot=varImpPlot+labs(x = "")
varImpPlot=varImpPlot+xlab("Factors/Variables")
varImpPlot=varImpPlot+ylab("Importance")
varImpPlot=varImpPlot+theme_bw(base_size = 12, base_family = "")
varImpPlot=varImpPlot+theme(legend.position="top", legend.background = element_rect(colour = "black"), plot.title = element_text(size = rel(1.5), colour = "black"), axis.text.y  = element_text(size=rel(1.5)), axis.text.x  = element_text(angle=90, vjust=0.5, size=rel(1.5)), panel.background = element_rect(fill = "white"), panel.grid.major = element_line(colour = "grey40"), panel.grid.minor = element_blank())
print(varImpPlot)

boxplot(access ~ group, data = final_df)

##transformed
m=ggplot(final_df,aes(x=c3,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

RFtransdata=final_df
names(final_df)
#RFtransdata[,2:dim(RFtransdata)[2]]<- lapply(2:dim(RFtransdata)[2], function(x) log10(ifelse(RFtransdata[,x]<0,0,RFtransdata[,x])+1))
RFtransdata[,2:dim(RFtransdata)[2]]<- lapply(2:dim(RFtransdata)[2], function(x) sqrt(RFtransdata[,x]))

m=ggplot(RFtransdata,aes(x=access,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

# Look at how transformations changed data
boxplotdata=RFtransdata[,c(2:dim(RFtransdata)[2])]
par(mfrow=c(3,3))
for (i in 1:length(boxplotdata)) {
  boxplot(boxplotdata[,i], main=names(boxplotdata[i]))
}


####nnet importance
#extract best ANN model parameters 
#### libraries ####
pckg = c("data.table","devtools","caret","NeuralNetTools",
         "nnet","reshape","dplyr","ggplot2","gridExtra",
         "grid") 

usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

lapply(pckg,usePackage)
##### end libraries ####

bar.theme_nolegend = theme_classic(base_size = 12, base_family = "Arial") + theme(plot.title = element_text(face="bold", colour="#000000", size=14, vjust=1.3), axis.title.x = element_text(face="plain", colour="#000000", size=15, vjust=-0.3),
                                                                                  axis.text.x  = element_text(angle=90, vjust=0.5, size=9), axis.title.y = element_text(face="plain", colour="#000000", size=15, vjust=0.8), axis.text.y  = element_text(angle=0, vjust=0.5, size=12)) + 
  theme(plot.background = element_blank(),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),panel.border = element_blank()) + 
  theme(axis.line = element_line(color = 'black')) + theme(legend.position = "none")

plots.name = NULL
for (pattern in levels(final_df$group)){
  plot.temp = olden(ann$finalModel, pattern) + bar.theme_nolegend + theme(legend.position = 'none')
  plot.temp = plot.temp + geom_bar(colour="black", fill="gray", stat="identity") +
    annotate("text", x=7, y=1, label=pattern, color="red",
             angle = 0)
  
  #save plot
  assign(paste("g_",pattern,sep=""), ggplotGrob(plot.temp))
  rm(plot.temp)
}

plots.sen = list.files(pattern = "g_*")

png(file = paste(analysis.plots.path,"/","ANNs_sensitivity","_",dataset,"_",wsize,".png",sep=""), width = 900, height = 600)
grid.arrange(g_1, g_2, g_3, g_4, g_5, g_6, ncol=3)
dev.off()


print(plot.temp)
