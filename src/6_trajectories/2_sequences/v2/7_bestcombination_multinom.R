library(broom)

Ord_Logit_Auto <- function(Dataset,D_Var,Ind_Var,Dev_Val_Split_Per=25,Imputation=TRUE,Imputation_Per=5,Outlier_Treatment=TRUE){
  ####Required Libraries
  wants <- c("moments", "DMwR", "Boruta", "car", "gtools","aod","VGAM")
  has   <- wants %in% rownames(installed.packages())
  if(any(!has)) install.packages(wants[!has])
  
  library(moments)
  library(DMwR)
  library(Boruta)
  library(car)
  library(gtools)
  library(aod)
  library(VGAM)
  
  ####Summary Statistics Computation
  summary_stats <- function(x)
  {return(c(
    nrow(Dataset),nrow(Dataset)-NROW(na.omit(x)),min(x,na.rm=T),quantile(x,c(.01,.05,.10,.25,.50,.75,.9,.95,.99),na.rm=T),
    max(x,na.rm=T),mean(x,na.rm=T),median(x,na.rm=T),sd(x,na.rm=T),100*sd(x,na.rm=T)/mean(x,na.rm=T),max(x,na.rm=T)- min(x,na.rm=T),
    IQR(x,na.rm=T),skewness(x,na.rm=T),kurtosis(x,na.rm=T)
  ))}
  Summary_Data <- as.data.frame(apply(Dataset[,c(Ind_Var)], 2, summary_stats))
  Summary_Data$stat <-c("OBS","NMIS","0% Min","1%","5%","10%","25% Q1","50% Median","75% Q3","90%","95%","99%","100% Max","Mean","Median","SD","CV (Mean/SD)","Range (Max-Min)","IQR (Q3-Q1)","Skewness(-1,+1)","Kurtosis(0,1)")
  Summary_Data<-data.frame(Summary_Data, row.names=NULL)
  Summary_Data <- cbind(Summary_Data$stat,Summary_Data[,(names(Summary_Data) != "stat")] )
  names(Summary_Data)[names(Summary_Data) == "Summary_Data$stat"] <- "Stats"
  Summary_Data[,-1] <-round(Summary_Data[,-1],2) 
  Univar_Stats_Pre_Outlier <- Summary_Data
  
  #### Missing Value Imputation
  #complete.cases(Dataset[,c(D_Var,Ind_Var)])
  imp_Flag <- sum(as.numeric(Univar_Stats_Pre_Outlier[2,-1]) > 0) > 0
  if (imp_Flag ==TRUE){
    if (Imputation == TRUE){
      Out_Names <- colnames(Summary_Data)[which(Summary_Data[2,2:ncol(Summary_Data)] > Imputation_Per)+1]
      Dataset <- Dataset[,!names(Dataset) %in% Out_Names]
      Dataset <- knnImputation(Dataset)
    }
  }
  
  #### Outlier Treatment
  a <- colnames(Dataset)[colnames(Dataset) %in% c(D_Var,Ind_Var)]
  if (Outlier_Treatment == TRUE){
    for (i in c(1:length(a))){
      if (length(unique(Dataset[,a[i]])) > 20) {
        Dif_95_99  <- round(((Summary_Data[12,a[i]]-Summary_Data[11,a[i]])/Summary_Data[11,a[i]]),3)
        Dif_99_100 <- round(((Summary_Data[13,a[i]]-Summary_Data[12,a[i]])/Summary_Data[12,a[i]]),3)
        if (Dif_95_99  >= .25){Dataset[(Dataset[,a[i]] > Summary_Data[11,a[i]]),a[i]] <- Summary_Data[11,a[i]]}
        if (Dif_99_100 >= .25){Dataset[(Dataset[,a[i]] > Summary_Data[12,a[i]]),a[i]] <- Summary_Data[12,a[i]]}
      }
    }
  }
  
  #Summary Statistics Post Imputation & Outlier Treatment
  Summary_Data <- as.data.frame(apply(Dataset[,colnames(Dataset) %in% c(Ind_Var)], 2, summary_stats))
  Summary_Data$stat <-c("OBS","NMIS","0% Min","1%","5%","10%","25% Q1","50% Median","75% Q3","90%","95%","99%","100% Max","Mean","Median","SD","CV (Mean/SD)","Range (Max-Min)","IQR (Q3-Q1)","Skewness(-1,+1)","Kurtosis(0,1)")
  Summary_Data<-data.frame(Summary_Data, row.names=NULL)
  Summary_Data <- cbind(Summary_Data$stat,Summary_Data[,(names(Summary_Data) != "stat")] )
  names(Summary_Data)[names(Summary_Data) == "Summary_Data$stat"] <- "Stats"
  Summary_Data[,-1] <-round(Summary_Data[,-1],2)  
  Univar_Stats_Post_Outlier <- Summary_Data
  
  #### Check for Perfect MultiCollinearity
  Dataset_1 <- cbind(Dataset,D_Var_1 = as.numeric(Dataset[,D_Var]))
  D_Var_1 <- "D_Var_1"
  
  varlist <- colnames(Dataset)[colnames(Dataset) %in% Ind_Var]
  Formula <- formula(paste(paste(D_Var_1,"~ "), paste(varlist, collapse=" + ")))
  fit <- lm(Formula, data=Dataset_1)
  n <- rownames(alias(fit)$Complete)
  tempNames <- varlist[!varlist %in% n]
  
  ####   Variable Reduction Using Baruta
  Boruta(formula(paste0(D_Var,"~.")),data=Dataset[,c(D_Var,tempNames)],doTrace=2)->Bor.son;
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
  
  set.seed(1000)
  inTraining <- createDataPartition(Dataset$group, p = .80, list = FALSE, times = 1)
  
  Dataset_D  <- Dataset[inTraining,]
  Dataset_V <- Dataset[-inTraining,]
  
  ##### Defining All (2^n-1) Possible Combinations of Selected Variables
  for (m in c(1:length(tempNames))){
    y <- combinations(length(tempNames),m,tempNames,repeats=FALSE)
    
    for (n in c(1:length(y[,1]))){
      y1 <- c(y[n,])
      Var_List <- paste(y1,collapse=" + ")
      
      Formula <- formula(paste(paste(D_Var,"~ "), paste(y1, collapse=" + ")))
      vglmFit <- multinom(Formula, data = Dataset_D)
      
      Pred_vglmFit <- predict(vglmFit, Dataset_V, type="class")
      
      OA =confusionMatrix(Dataset_V$group, Pred_vglmFit)$overall["Accuracy"]

      #AIC of the Model
      AIC <- round(AIC(vglmFit),2)
      print(AIC)
      BIC <- round(BIC(vglmFit),2)
      sumOrd   <- summary(vglmFit)
      X_Reg <- as.data.frame(cbind(OA=OA,AIC=AIC,BIC=BIC,Var_list=Var_List))
      ifelse(n==1 ,X_Reg_1 <- X_Reg, X_Reg_1 <-rbind(X_Reg_1,X_Reg))
    }
    ifelse(m==1 ,All_Models <- X_Reg_1, All_Models<-rbind(All_Models,X_Reg_1))
  }
  #All_Models <- subset(All_Models, significance=="Sig")
  return (list(All_Models,Univar_Stats_Pre_Outlier,Univar_Stats_Post_Outlier))
}

Output <- Ord_Logit_Auto(final_df,"group",c(explanatory_ext,explanatory_int))
All_Models <- as.data.frame(Output[1])
All_Models <-All_Models[order(All_Models$AIC, All_Models$BIC),]

explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatoryv2")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)

file_name <- paste0('bm_multinom_combinations..RSav')
file_path <- paste0(explanatory_dir,'/',file_name)

save(All_Models, file=paste0(file_path))


Univar_Stats_Pre_Outlier <- as.data.frame(Output[2])
Univar_Stats_Post_Outlier <- as.data.frame(Output[3])
