rm(list = ls())

library(TraMineR)
library(finalfit)
library(labelled)
library(raster)
library(tibble)
library(dplyr)
library(ggplot2)
library(JLutils)

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

access <- raster(paste0(aux_dir,'/ancillary/processed/access.tif'))
dem <- raster(paste0(aux_dir,'/ancillary/processed/srtm.tif'))
slope <- raster(paste0(aux_dir,'/ancillary/processed/slope.tif'))
precipitation <- raster(paste0(aux_dir,'/ancillary/processed/bio12.tif'))
pas_conservation <- raster(paste0(aux_dir,'/ancillary/processed/distance_PAs_conservation_AMZ.tif'))
pas_exploitation <- raster(paste0(aux_dir,'/ancillary/processed/distance_PAs_exploitation_AMZ.tif'))

###start process
fn <- 'WARD_clustersk6_2004-2016_minperiod12_TRATE_OM_target.RSav'

file_name <- paste0('auxdata_',fn)
file_path <- paste0(explanatory_dir,'/',file_name)

if(file.exists(paste0(file_path))){
  load(file_path)
} else{
  load(paste0(clusters_dir,'/',fn))
  
  ##extract raster values
  ##add and prepare aux columns
  tab.target_geo = cluster_all
  coordinates(tab.target_geo) <- c("x", "y")
  mypoints = SpatialPoints(tab.target_geo,proj4string = CRS("+init=epsg:4326"))
  
  dem_val =raster::extract(dem, mypoints)
  slope_val =raster::extract(slope, mypoints)
  access_val =raster::extract(access, mypoints)
  prec_val =raster::extract(precipitation, mypoints)
  pascon_val =raster::extract(pas_conservation, mypoints)
  pasexp_val =raster::extract(pas_exploitation, mypoints)
  
  tab.target_geo$access = (access_val)/(60*24)
  tab.target_geo$dem = dem_val
  tab.target_geo$slope = slope_val
  tab.target_geo$prec = prec_val
  tab.target_geo$pascon = (pascon_val * 231.91560544825498) / 1000
  tab.target_geo$pasexp = (pasexp_val * 231.91560544825498) / 1000
  
  save(tab.target_geo, file=paste0(file_path))
}

## Create categorical data from covariates
tab.target_geo$accessK2<-cut(tab.target_geo$access, c(0,0.5,max(tab.target_geo$access, na.rm = TRUE)), labels = c("Half day","> Half day"))
tab.target_geo$demK2<-cut(tab.target_geo$dem, c(0,250,max(tab.target_geo$dem, na.rm = TRUE)),labels = c("<250m",">=250m"))
tab.target_geo$slopeK2<-cut(tab.target_geo$slope, c(-1,6,max(tab.target_geo$slope, na.rm = TRUE)), labels = c("Flat","Steep"))
tab.target_geo$precK2<-cut(tab.target_geo$prec, c(0,2000,max(tab.target_geo$prec, na.rm = TRUE)),labels = c("<2000mm",">=2000m"))
tab.target_geo$pasconK2<-cut(tab.target_geo$pascon, c(0,50,max(tab.target_geo$pascon, na.rm = TRUE)),labels = c("<50km",">=50km"))
tab.target_geo$pasexpK2<-cut(tab.target_geo$pasexp, c(0,50,max(tab.target_geo$pasexp, na.rm = TRUE)),labels = c("<50km",">=50km"))

group <- factor(
  tab.target_geo$clusters,
  c(1, 2, 3, 4, 5, 6),
  c("1", "2", "3", "4", "5", "6")
)

tab.target_df = as.data.frame(tab.target_geo)

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
  accessK2 + slopeK2 ~ group,
  data = to_factor(final_df),
  label = "scales::percent(col.prop)",
  breaks = c(-Inf, -4, -2, 0, 2, 4, Inf)
)

###multinomial
library(nnet)
final_df$group2 <- relevel(final_df$group, 1)
regm <- multinom(
  group2 ~ accessK2 + slopeK2, 
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
  xlab("Facteurs") + ylab("Odds Ratios") +
  labs(color = "vs. Water") +
  theme_light() + 
  theme(panel.grid.major.y = element_blank())


library(ggeffects)
cowplot::plot_grid(plotlist = plot(ggeffect(regm)), ncol = 2)


