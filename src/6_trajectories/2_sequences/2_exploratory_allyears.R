# Working directory and libraries
library(TraMineR)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(extrafont)

tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
chart_dir <- paste0("E:/acocac/research/",tile,"/trajectories/charts_allyears")
dir.create(chart_dir, showWarnings = FALSE, recursive = T)

##analysis with simple classes
targetyears = c(2004:2018)
lc_target_years <-c(2001,2019)

datalist = list()

for (i in 1:length(targetyears)){
  infile <- paste0(indir,'/','resultsseq_tyear_',as.character(targetyears[i]),'_simple.rdata')
  results_df = rlist::list.load(infile)
  
  tab.stable = results_df[1][[1]]
  tab.nonstable = results_df[2][[1]]

  tab.target.seq = tab.nonstable
  
  tab.target.metrics = tab.target.seq
  
  stat.bf <- seqstatd(tab.target.metrics)
  ent <- stat.bf$Entropy
  
  dat <- data.frame(ent)
  dat$year <-targetyears[i]
  dat$time <-names(ent)
  
  datalist[[i]] <- dat # add it to your list
  }

df <- data.table::rbindlist(datalist)

df$year <- as.character(df$year)

iris_new <- df %>%
  mutate(ref = if_else(year == time, ent, 0))

bigfoo <- subset(iris_new, ref > 0) #use only those values that are larger than 0

png(filename=paste0(chart_dir,"/tentropy_",paste(targetyears[1],tail(targetyears, n=1),sep='-'),".png"), width = 200, height = 200, units='mm', res=100)
ggplot(df, aes(time, ent, group = year, colour=year)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(data=bigfoo, mapping = aes(time, ref, group = year),color = "steelblue", size = 3, shape=4, stroke = 2) +
  labs(title = "Entropy per Terra-i's deforestation year",
       subtitle = "All LC years available (2001-2019)",
       y = "Landscape Entropy [0-1]", x = "") + 
  facet_wrap(~ year, ncol = 3) +
  theme_tufte() + theme(axis.line=element_line()) + 
  scale_y_continuous(limits=c(0,1)) +
  theme(axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + 
  theme(text=element_text(size=13,  family="sans"))

dev.off()

