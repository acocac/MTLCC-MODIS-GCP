rm(list = ls())

library(TraMineR)
library(nnet)
library(stargazer)
library(doBy)
library(ggeffects)
library(ggplot2)
library(ggfortify)
library(vegan)
library(WeightedCluster)

tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatory")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)
clusters_dir <- paste0("E:/acocac/research/",tile,"/trajectories/clusters")
dir.create(clusters_dir, showWarnings = FALSE, recursive = T)
seqdist_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_distances")
dir.create(seqdist_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_data")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)


fn <- 'db_WARD_clustersk8_2004_TRATE_OM.rdata'
load(paste0(explanatory_dir,'/',fn))
aux_unique <- x[[2]]

fn <- '2004_TRATE_OM.RSav'
load(paste0(seqdist_dir,'/',fn))

fn <- '2004_unique_seq.RSav'
load(paste0(seqdata_dir,'/',fn))
seq_weights <- attr(seq_object,"weights")

aux_unique$slopeK2 <-cut(aux_unique$slope, c(-1,6,max(aux_unique$slope, na.rm = TRUE)))

da1 <- dissassoc(seq_dist, weights=seq_weights, group = aux_unique$slopeK2, R = 50, weight.permutation="diss")
print(da1$stat)

dt<- disstree(seq_dist ~ male +cohort1 +cohort2 +cohort3 +cohort4 +red +black +fedu +medu +primary +middle +high +minor +party+ loc1+ loc2+ loc3+ loc4+ loc5+ loc6, data=all, R=100) #here the dummy variables are forbidden


##weighted MDS
mds2 = wcmdscale(seq_dist, k=2, w=seq_weights)
# plot
plot(mds2[,1], mds2[,2], type = "n", xlab = "", ylab = "",
     axes = FALSE, main = "wcmdscale (vegan)")
text(mds2[,1], mds2[,2], labels(seq_dist), cex = 0.9, xpd = TRUE)


mvad$test <- rep(-1, nrow(mvad))
for(clust in unique(pamclust4$clustering)){
  cond <- pamclust4$clustering == clust
  values <- worsq[cond, 2]
  mvad$test[cond] <- as.integer(values > weighted.median(values, w=mvad$weight[cond]))
}
mvad$test <- factor(mvad$test, levels=0:1, labels=c("non-test", "test"))




parent.mds <- cmdscale(seq_dist,k=2,eig=TRUE)

autoplot(parent.mds, label = TRUE, label.size = 3)



# plot solution
x <- parent.mds$points[,2]
y <- parent.mds$points[,3]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="Metric MDS", type="n")
text(x, y, labels = row.names(parent.mds), cex=.7)

plot(parent.mds)


aux_target$accessK2<-cut(aux_target$access, c(0,0.5,max(aux_target$access, na.rm = TRUE)))
aux_target$slopeK2<-cut(aux_target$slope, c(-1,6,max(aux_target$slope, na.rm = TRUE)))
aux_target$demK2<-cut(aux_target$dem, c(0,250,max(aux_target$dem, na.rm = TRUE)))
aux_target$precK2<-cut(aux_target$prec, c(0,2000,max(aux_target$prec, na.rm = TRUE)))
aux_target$pasconK2<-cut(aux_target$pascon, c(0,50,max(aux_target$pascon, na.rm = TRUE)))
aux_target$pasexpK2<-cut(aux_target$pasexp, c(0,50,max(aux_target$pasexp, na.rm = TRUE)))


###stats by cluster
stats_aux <-(summaryBy(access+dem+slope+prec+pascon+pasexp ~ clusters, 
                  data=as.data.frame(aux_target),FUN=c(mean,sd),na.rm=TRUE))

mult_all <- multinom(clusters ~ access+dem+slope+prec+pascon+pasexp, 
                  data = aux_target)

summary(mult4)

stargazer(mult4, no.space = T, 
          comult_allvariate.labels = c("access", "dem", "slope", "prec", "pascon","pasexp"))

mult1 <- multinom(clusters ~ slopeK2, 
                  data = aux_target)

mult1 <- multinom(clusters ~ slope, 
                  data = aux_target)

summary(mult1)

stargazer(mult1, no.space = T, 
          covariate.labels = c("slopebin"))

sexpp <- ggeffect(mult1, terms = c("slopeK2"))

colnames(sexpp)[colnames(sexpp) == 'response.level'] <- 'class'
sexpp$class <- as.factor(sexpp$class)

# Revalue the gender factors to be readable
sexpp$x <- as.factor(sexpp$x)
levels(sexpp$x)[levels(sexpp$x)=="(-1,6]"] <- "Flat"
levels(sexpp$x)[levels(sexpp$x)=="(6,48.6]"] <- "Step"

sexpp$x <- as.factor(sexpp$x)
levels(sexpp$x)[levels(sexpp$x)=="(0,0.5]"] <- "Near"
levels(sexpp$x)[levels(sexpp$x)=="(0.5,3.31]"] <- "Far"


fig4 <- sexpp %>%
  ggplot(aes(x, predicted, fill = x, label = round(predicted, 0))) +
  geom_col() +
  facet_grid(~class) +
  theme_minimal() +
  ggtitle("Figure 4. Sequence Cluster by Gender") +
  labs(x = NULL, y = NULL, subtitle = "Predicted proportion per cluster with model controls",
       caption = "Source: American Time Use Surveys (2017) \n Models control for education, race-ethnicity, marital status, extra adults,
       number of household kids, kids under 2, age, weekend diary day") +
  theme(plot.subtitle = element_text(size = 11, vjust = 1),
        plot.caption  = element_text(vjust = 1, size =8, colour = "grey"), 
        legend.position="none",
        strip.text.x  = element_text(size = 8, face = "bold"),
        axis.title    = element_text(size = 9), 
        axis.text     = element_text(size = 8), 
        plot.title    = element_text(size = 12, face = "bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank())

fig4
