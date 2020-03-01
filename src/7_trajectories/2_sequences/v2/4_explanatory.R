rm(list = ls())

library(TraMineR)
library(finalfit)
library(labelled)
library(raster)
library(tibble)
library(dplyr)
library(ggplot2)
library(JLutils)
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
  tab.target_geo = cluster_all
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

## Create categorical data from covariates
tab.target_geo$accessK2<-cut(tab.target_geo$access, c(0,0.5,max(tab.target_geo$access, na.rm = TRUE)), labels = c("Half day","> Half day"))
tab.target_geo$demK2<-cut(tab.target_geo$dem, c(0,250,max(tab.target_geo$dem, na.rm = TRUE)),labels = c("<250m",">=250m"))
tab.target_geo$slopeK2<-cut(tab.target_geo$slope, c(-1,6,max(tab.target_geo$slope, na.rm = TRUE)), labels = c("Flat","Steep"))
tab.target_geo$precK2<-cut(tab.target_geo$prec, c(0,2000,max(tab.target_geo$prec, na.rm = TRUE)),labels = c("<2000mm",">=2000m"))
tab.target_geo$pasconK2<-cut(tab.target_geo$pascon, c(0,50,max(tab.target_geo$pascon, na.rm = TRUE)),labels = c("Prox cPAs <50km","Prox cPAs >=50km"))
tab.target_geo$pasexpK2<-cut(tab.target_geo$pasexp, c(0,50,max(tab.target_geo$pasexp, na.rm = TRUE)),labels = c("Prox ePAs <50km","Prox ePAs >=50km"))

group <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("1", "2", "3", "4", "5", "6")
)

tab.target_df2 = as.data.frame(tab.target_geo)
rm(tab.target_geo)
final_df = data.frame(group, tab.target_df[,23:28])
final_df = final_df[complete.cases(final_df),]
names(final_df)
#
# var_label(final_df$accessK2) <- "Accesibility"
# val_labels(final_df$accessK2) <- c(halfday = "0", gthalfday = 1)
# var_label(final_df$slopeK2) <- "Slope"
# val_labels(final_df$slopeK2) <- c(Flat = 0, Steep = 1)

explanatory <- c('accessK2','demK2','slopeK2','precK2','pasconK2','pasexpK2')
dependent <- "group"
tab <- summary_factorlist(
  to_factor(final_df), dependent, explanatory,
  p=TRUE, column = TRUE, total_col = TRUE
)
knitr::kable(tab, row.names = FALSE)

res <- tibble()
explanatory <- c(
  "accessK2" = "Accesibility",
  "slopeK2" = "Slope",
  "demK2" = "Elevation",
  "precK2" = "Precipitation",
  "pasconK2" = "Proximity to conservation PAs",
  "pasexpK2" = "Proximity to exploitation PAs"
)

for (v in names(explanatory)) {
  tmp <- tibble::as_tibble(table(final_df$group, to_factor(final_df[[v]])), .name_repair = "unique")
  names(tmp) <- c("group", "level", "n")
  test <- chisq.test(final_df$group, to_factor(final_df[[v]]))
  tmp$var <- paste0(
    explanatory[v],
    "\n",
    scales::pvalue(test$p.value, add_p = TRUE)
  )
  res <- bind_rows(res, tmp)
}

ggplot(data = res) +
  aes(x = level, fill = group, weight = n) +
  geom_bar(position = "fill") +
  JLutils::stat_fill_labels() +
  facet_grid(var ~ ., scales = "free", space = "free") +
  scale_y_continuous(labels = scales::percent, breaks = 0:5/5) +
  coord_flip() +
  theme(legend.position = "bottom") +
  xlab("") + ylab("") + labs(fill = "")

ggchisq_res(
  accessK2 + demK2 + slopeK2 + precK2 + pasconK2 + pasexpK2 ~ group,
  data = to_factor(final_df),
  label = "scales::percent(col.prop)",
  breaks = c(-Inf, -4, -2, 0, 2, 4, Inf)
)

###multinomial
library(nnet)
final_df$group2 <- relevel(final_df$group, 1)
regm <- multinom(
  group2 ~ accessK2 + demK2 + slopeK2 + precK2 + pasconK2 + pasexpK2,
  data = to_factor(final_df)
)

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
  labs(color = "vs. 1 - DF>W") +
  theme_light() +
  theme(panel.grid.major.y = element_blank())


library(ggeffects)
cowplot::plot_grid(plotlist = plot(ggeffect(regm)), ncol = 2)

###analysis iabd
tab.target_df = as.data.frame(tab.target_geo)

group <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("1", "2", "3", "4", "5", "6")
)

final_df = data.frame(group, tab.target_df[,17:22])
names(final_df)
final_df = final_df[complete.cases(final_df),]

var_0=final_df[final_df[,"group"]!=0,"access"]
var_1=final_df[final_df[,"group"]==1,"access"]

min_sample = length(var_1)

var_pres=cbind(c(sample(var_0,min_sample),var_1),c(array("other",min_sample),array("group1",min_sample)))
var_pres=data.frame(c(sample(var_0,min_sample),var_1),c(array("other",min_sample),array("group1",min_sample)))
names(var_pres)=c("var_pres","group_target")

model2=glm(group_target~var_pres,family="binomial",data=var_pres)
summary(model2)



m=ggplot(min_pres,aes(x=min_pres_dist,group=group_target,fill=group_target,alpha=1.5))
m+geom_density(size=1)+geom_text(aes(0.5,5,label = "pvalue=0.07"),colour="black",size=8)+scale_alpha(guide = 'none')
names(final_df)

m=ggplot(final_df,aes(x=access,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=dem,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=slope,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=prec,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=pasexp,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

m=ggplot(final_df,aes(x=pascon,group=group,fill=group,alpha=1.5))
m+geom_density(size=1)+scale_alpha(guide = 'none')

#####analysis based on http://www.css.cornell.edu/faculty/dgr2/teach/R/R_lcc.pdf###
###binomial
d2 <- function(model) { round(1-(model$deviance/model$null.deviance),4) }
d2(model2)

#The logistic model of change for continuous predictors
summary(final_df$access); summary(final_df$slope)

par(mfrow=c(1,2))
hist(final_df$access); hist(final_df$slope)
par(mfrow=c(1,1))

##variables transpformation
ds.l<-log(final_df$access+30); dr.l<-log(final_df$slope+30)
par(mfrow=c(1,2))
hist(ds.l); hist(dr.l)
par(mfrow=c(1,1))

##variables correlation
plot(dr.l, ds.l)
abline(h=mean(ds.l),lty=2); abline(v=mean(dr.l),lty=2)
cor.test(dr.l,ds.l)

par(mfrow=c(1,2))
plot(access~group, data=final_df)
plot(slope~group, data=final_df)
par(mfrow=c(1,1))

lm.access <- lm(access~group, data=final_df); summary(lm.access)

###multiomial
final_df$group2 <- relevel(final_df$group, 1)
mlm.t.drs <- multinom(
  group2 ~ access + pasexp,
  data = to_factor(final_df)
)

summary(mlm.t.drs, Wald=T)

range(final_df$access)

to.predict <- expand.grid(group_traj = levels(final_df$group),
                          access = seq(min(range(final_df$access)), max(range(final_df$access)), by=.2),
                          pasexp = seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50))

p.fit <- predict(mlm.t.drs, to.predict, type="probs")
head(p.fit, 10)

to.graph <- subset(to.predict, (group_traj=="2") & (access == 3))

g.fit <- predict(mlm.t.drs, to.graph, type="probs")

par(mfrow=c(1,1))
plot(c(min(range(final_df$pasexp)), max(range(final_df$pasexp))), c(0,1), type="n")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"1"], lty=1, lwd=3, col="green")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"2"], lty=2, lwd=3, col="magenta")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"3"], lty=3, lwd=3, col="red")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"4"], lty=4, lwd=3, col="blue")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"5"], lty=5, lwd=3, col="black")
lines(seq(min(range(final_df$pasexp)), max(range(final_df$pasexp)), by=50), g.fit[,"6"], lty=6, lwd=3, col="gray")
# text(4, .9, "No deforestation", col="green", pos=4)
# text(4, .85, "Partial deforestation", col="magenta", pos=4)
# text(4, .8, "Complete deforestation", col="red", pos=4)


###reference approach #continuos
explanatory <- c('access','dem','slope','prec','pascon','pasexp')
dependent <- "group"
tab <- summary_factorlist(
  to_factor(final_df), dependent, explanatory,
  p=TRUE, column = TRUE, total_col = TRUE
)
knitr::kable(tab, row.names = FALSE)

#multinomial
final_df$group2 <- relevel(final_df$group, 1)
regm <- multinom(
  group2 ~ access + dem + slope + prec + pascon + pasexp,
  data = to_factor(final_df)
)

summary(regm, Wald=T)
head(fitted(regm))

head(final_df$group)

p.fit <- predict(regm, final_df, type="probs")

prob_df <- data.frame(unique_seq_subset[,1:2],p.fit)

xyz <- as.data.frame(cbind(prob_df$x,prob_df$y,prob_df$X1))

names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_cluster <- raster(xyz)
crs(raster_cluster) <- CRS('+init=EPSG:4326')

#required libraries
list.of.packages <- c("caret","digest","doParallel","foreach","foreign","fpc","ggplot2","gtools","GWmodel","jpeg","mclust","NbClust","parallel","plyr","pracma","ranger","raster","reshape","raster","rgdal","rgeos","scales","spdep","spgwr","stringr","tmap","zoo")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)){install.packages(new.packages)}
lapply(list.of.packages, require, character.only = TRUE)

data("deforestation")
tmap_mode("view")
tm_basemap("OpenStreetMap") +
  tm_shape(raster_cluster) +
  tm_raster(style = "cat")

#tm_raster(style = "cat") +


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


library(ggeffects)
cowplot::plot_grid(plotlist = plot(ggeffect(regm)), ncol = 2)
