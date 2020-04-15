rm(list = ls())

library(TraMineR)
library(nnet)
library(stargazer)
library(doBy)
library(ggeffects)
library(ggplot2)


tile <- 'AMZ'
lc_target_years <-c(2001,2019)

#dirs
explanatory_dir <- paste0("E:/acocac/research/",tile,"/trajectories/explanatory")
dir.create(explanatory_dir, showWarnings = FALSE, recursive = T)

fn <- 'db_WARD_clustersk8_2004_TRATE_OM.rdata'
load(paste0(explanatory_dir,'/',fn))
aux_target <- x[[1]]

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
