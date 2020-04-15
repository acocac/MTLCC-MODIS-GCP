rm(list = ls())

library(data.table)
library(labelled)
library(questionr)
library(viridis)
library(tidyr)
library(TraMineR, quietly = TRUE)

##settings
tile <- 'AMZ'

#dirs
indir <- paste0("E:/acocac/research/",tile,"/trajectories/data")
descriptive_dir <- paste0("E:/acocac/research/",tile,"/trajectories/descriptivev2")
dir.create(descriptive_dir, showWarnings = FALSE, recursive = T)
seqdata_dir <- paste0("E:/acocac/research/",tile,"/trajectories/sequence_datav2")
dir.create(seqdata_dir, showWarnings = FALSE, recursive = T)
charts_dir <- paste0("E:/acocac/research/",tile,"/trajectories/chartsv2")
dir.create(charts_dir, showWarnings = FALSE, recursive = T)

global_classes <- c('Barren',
                    'Water Bodies',
                    'Urban and Built-up',
                    'Dense Forests',
                    'Open Forests',
                    'Farmland',
                    'Shrublands')

global_shortclasses <- c('Ba', 
                         'W', 'Bu',
                         'DF', 'OF', 
                         'Fm',
                         'S')
global_colors = c('#f9ffa4', 
                  '#1c0dff', '#fa0000', 
                  '#003f00', '#006c00', 
                  '#db00ff', 
                  '#dcd159') 


##implement descriptive##
targetyears = c(2004:2016)

file_name <- paste0(min(targetyears),'-',max(targetyears),'_target_tb')
load(paste0(seqdata_dir,'/',file_name,'.RSav'))

#reshape
target_reshape <- melt(target_tb, id = c("x","y","tyear"), value.name = "LC")
names(target_reshape) = c("x","y","tyear","lyear","LC")
target_reshape$lyear <- as.numeric(as.character(target_reshape$lyear))

#describe(target_reshape, freq.n.max = 10)

target_reshape <- target_reshape[complete.cases(target_reshape$LC), ] 
classes <- unique(target_reshape$LC)

target_short <- global_shortclasses[sort(classes)]
target_colors <- global_colors[sort(classes)]
target_long <- global_classes[sort(classes)]



##plot change by year
ggplot(target_reshape) +
  aes(x = lyear) +
  geom_bar()

#status
n <- target_reshape[lyear %in% (0:17), .(n = .N), by = lyear]$n
etiquettes <- paste0("Y", 1:18, "\n(", round(n*10^-6,1), "M)")
val_labels(target_reshape$LC) <- c(
  "DF" = 4,
  "OF" = 5,
  "Fm" = 6,
  "W" = 2,
  "Bu" = 3
)

ggplot(target_reshape) +
  aes(x = lyear, fill = to_factor(LC)) +
  geom_bar(color = "gray50", width = 1) +
  scale_x_continuous(breaks = 0:17, labels = etiquettes) +
  theme(axis.text.x=element_text(angle=45,hjust=1)) +
  ggtitle("Distribution of LC types following deforestation") +
  xlab("") + ylab("") +
  theme_light() +
  theme(legend.position = "bottom") +
  labs(fill = "Land cover") + 
  scale_fill_viridis(discrete = TRUE, direction = -1) +
  guides(fill = guide_legend(nrow = 1))

###Évolution de la cascade de soins au cours du temps
ggplot(target_reshape) +
  aes(x = lyear, fill = to_factor(LC)) +
  geom_bar(color = "gray50", width = 1, position = "fill") +
  scale_x_continuous(breaks = 0:17, labels = etiquettes) +
  scale_y_continuous(labels = scales::percent) +
  ggtitle("Proportion of LC types following deforestation") +
  xlab("") + ylab("") +
  theme_light() +
  theme(legend.position = "bottom") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5)) +
  labs(fill = "LC types") + 
  #scale_colour_manual(values = target_colors) +
  scale_fill_manual(values = target_colors,
                    breaks = classes,
                    labels = target_short) +
  guides(fill = guide_legend(nrow = 1))


target_reshape$LC <- factor(target_reshape$LC, levels=sort(classes), ordered=TRUE)

png(file = paste0(charts_dir,"/LCpartitionDF_2004-2018.png"), width = 850, height = 450, res = 90)
  ggplot(target_reshape) +
  aes(x = lyear, fill = to_factor(LC)) +
  geom_bar(color=NA, width = 1, position = "fill") +
  theme_ipsum_rc(axis_title_size = 16, base_size=9) +
  theme(legend.position="bottom") +
  scale_x_continuous(breaks = 0:17, labels = etiquettes) +
  scale_y_continuous(labels = scales::percent) +
  labs(x="Year following deforestation", y="Proportion") +
  # labs(x="Year following deforestation", y="Proportion",
  #      title="Proportion of LC types following deforestation",
  #      subtitle="Organised by year incl. total sequences in million (M)") +
  theme(legend.text=element_text(size=12)) +
  theme(panel.grid = element_blank(),
        panel.border = element_blank()) +
  scale_fill_manual("",values=target_colors,breaks=sort(classes),
                    labels=target_long) +
  guides(fill = guide_legend(nrow = 1))

dev.off()