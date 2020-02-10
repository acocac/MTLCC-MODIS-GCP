##### Analysis of High Temporal Resolution Land Use/Land Cover Trajectories
##### Supplementary material to article published in Land https://www.mdpi.com/journal/land
##### R script to carry out the analysis presented in the paper
##### JF Mas - Universidad Nacional Autónoma de México - jfmas@ciga.unam.mx


# Working directory and libraries
library(TraMineR)
library(raster)
library(fastcluster)
library(tidyr)
library(WeightedCluster)

dir <- "E:/acocac/research/fc1/post_fc70"
#dir <- "E:/acocac/research/fc2/eval/pred/2_inputs/ep5/bands/convgru64_02_fold0_19713"
pattern <- "MCD12Q1v6raw_LCType1_afterpost"

files <- list.files(path=dir, pattern="tif", all.files=FALSE, full.names=TRUE,recursive=TRUE)

files <- Filter(function(x) grepl(pattern, x), files)

serie <- raster::stack(files)

plot(serie[[1]])

####################################################################################???
## Elaboration of a table which describes for each pixel the land category at each time step + value of covariates

## using rasterToPoints(r) to get coordinates
tab <- as.data.frame(rasterToPoints(serie))

coodsls <- c("x","y")
yearls <- paste0("c",as.character(seq(2004,2017,1)))

names(tab) <- c(coodsls, yearls)

# A look at the first rows of the table:
head(tab)
# and a summary:
summary(tab)

# Put an extra column with the concatened sequence
tab2 <- unite(tab, sec, yearls, sep="-")
tab$sec <- tab2$sec

# Verify all the categories for all the time steps
clases <- c()
for (year in 3:(length(yearls)+2)){
  clases <- unique(c(clases,tab[,year]))
}
print(sort(clases))

### Determine the number of different sequences
unique_traj <- sort(unique(tab$sec))
length(unique_traj)  # 1651 different trajectories

# Determine the frequency of these trajectories
tab_freq <- as.data.frame(table(tab$sec))

# Sort the sequences from more to less frequent
tab_freq2 <- tab_freq[order(tab_freq$Freq,decreasing = TRUE),] 
head(tab_freq2)

## State sequence object
global_classes <- c('Evergreen needleleaf forest', 'Evergreen broadleaf forest',
                                 'Deciduous needleleaf forest', 'Deciduous broadleaf forest',
                                 'Mixed forest', 'Closed shrublands', 'Open shrublands',
                                 'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                                 'Croplands', 'Urban and built-up', 'Cropland natural vegetation mosaic',
                                 'Snow and ice', 'Barren or sparsely vegetated', 'Water')

global_shortclasses <- c('ENF', 'EBF',
                    'DNF', 'DBF',
                    'MF', 'CS', 'OS',
                    'WS', 'S', 'G', 'PW',
                    'C', 'Bu', 'CN',
                    'S', 'Ba', 'W')

global_colors = c('#05450a', '#086a10',
          '#54a708',
          '#78d203',
          '#009900',
          '#c6b044',
          '#dcd159',
          '#dade48',
          '#fbff13',
          '#b6ff05',
          '#27ff87',
          '#c24f44',
          '#a5a5a5',
          '#ff6d4c',
          '#69fff8',
          '#f9ffa4',
          '#1c0dff')

target_classes <- global_classes[sort(clases)]
target_short <- global_shortclasses[sort(clases)]
target_colors <- global_colors[sort(clases)]

alphabet <- sort(clases)

labels <- target_classes                             
short_labels <- target_short
palette <- target_colors
tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                  cpal = palette, labels = short_labels)

#######################################################################################
## Visualize the sequence data set 

# Plot 10 sequences in the tab.seq sequence object (chosen to show diversity)
some_seq <- tab.seq[c(19,5,973,976,34,84,930,3893,993,995),]
seqiplot(some_seq, with.legend = T, border = T, main = "Some sequences", legend.prop=0.2)

# Plot all the sequences in the data set, sorted by states from start.
#seqIplot(tab.seq, sortv = "from.start", with.legend = T, main = "Sequences 2004-2017")

# Plot the 10 most frequent sequences.
seqfplot(tab.seq, with.legend = T, main="Most common sequences", legend.prop=0.2)

#######################################################################################
## Explore the sequence data set by computing and visualizing descriptive statistics


# Compute and plot the state distributions by time step. 
# With border = NA, borders surrounding the bars are removed. 
seqdplot(tab.seq, with.legend = T, border = NA,main="Land cover (states) distribution", ylab="Proportion of study area", legend.prop=0.2)

# Compute and plot the transversal entropy index (Landscape entropy over time)
seqHtplot(tab.seq, main = "Entropy", ylab="Entropy index value",xlab=("Time"), legend.prop=0.2)

#Plot the sequence of modal states (dominant land cover) of the transversal state distributions.
seqmsplot(tab.seq, with.legend = T, main ="Most frequent land cover", legend.prop=0.2)

# Plot the mean time spent in each land cover category.
seqmtplot(tab.seq, with.legend = T, main = "Permanence", ylab="Number of 3 years periods", legend.prop=0.2)

### Longitudinal turbulence and entropy indices 
# Computed for each pixel over the time
tab$Turb <- seqST(tab.seq, norm=FALSE)
tab$Entrop <- seqient(tab.seq, norm=TRUE, base=exp(1))

# Generate rasters which represent these indices
xyt <- as.data.frame(cbind(tab$x,tab$y,tab$Turb))
names(xyt) <- c("x","y","t")
coordinates(xyt) <- ~ x + y
gridded(xyt) <- TRUE
raster_turb <- raster(xyt)
plot(raster_turb)

xye <- as.data.frame(cbind(tab$x,tab$y,tab$Entrop))
names(xye) <- c("x","y","e")
head(xye)
coordinates(xye) <- ~ x + y
gridded(xye) <- TRUE
raster_entrop <- raster(xye)
plot(raster_entrop)

# Calculate the correlation between both indices
cor(tab$Turb,tab$Entrop) # 0.9394874

## Computes the transition rates
tr_rates <- seqtrate(tab.seq)
print(tr_rates)

#######################################################################################
# Compute distances between sequences using different dissimilarity indices

## OM with substitution costs based on transition
## probabilities and indel set as half the maximum
## substitution cost
costs.tr <- seqcost(tab.seq, method = "TRATE",with.missing = FALSE)

#sol 1
ac <- wcAggregateCases(tab.seq)
uniqueMvad <- tab.seq[ac$aggIndex,1:14]
weight.seq <- seqdef(uniqueMvad, weights = ac$aggWeights)

#sol 2
set.seed(1)
sampled.indices <- sample(nrow(tab.seq), round(0.05*nrow(tab.seq)))
sampled.tab.seq <- tab.seq[sampled.indices, ]

seq_target <- sampled.tab.seq
df_target <- tab[sampled.indices, ]

dist.om1 <- seqdist(seq_target, method = "OM",indel = costs.tr$indel, sm = costs.tr$sm,with.missing = F)
dim(dist.om1)

### OM based on features
# tab_state_features <- data.frame(state=c(10,5,3,3,3,1))
# costs.gower <- seqcost(seq_target, method = "FEATURES",with.missing = FALSE,state.features = tab_state_features)
# print(costs.gower)
# 
# dist.om2 <- seqdist(tab.seq, method = "OM",indel = costs.gower$indel, sm = costs.gower$sm,with.missing = F)
# dim(dist.om2)

## LCS
dist.lcs <- seqdist(seq_target, method = "LCS")

## LCP
dist.lcp <- seqdist(seq_target, method = "LCP") 

# Elaboration a typology of the trajectories: build a Ward hierarchical clustering
# of the sequences from the different distances and retrieve for each cell sequence the
# cluster membership of the 5 class solution. 

## Cluster based on OM transition rates
clusterward_om1 <- hclust(as.dist(dist.om1),method="ward.D")
plot(clusterward_om1)
cl_om1 <- cutree(clusterward_om1, k = 5)
df_target$clusterom1 <- cl_om1
head(df_target)

## Cluster based on OM features
# clusterward_om2 <- hclust(as.dist(dist.om2),method="ward.D")
# plot(clusterward_om2)
# cl_om2 <- cutree(clusterward_om2, k = 5)
# df_target$clusterom2 <- cl_om2
# head(df_target)

## Cluster based on LCS
clusterward_lcs <- hclust(as.dist(dist.lcs),method="ward.D")
plot(clusterward_lcs)
cl_lcs <- cutree(clusterward_lcs, k = 5)
df_target$clusterlcs <- cl_lcs
head(df_target)

## Cluster based on LCP
clusterward_lcp <- hclust(as.dist(dist.lcp),method="ward.D")
plot(clusterward_lcp)
cl_lcp <- cutree(clusterward_lcp, k = 5)
df_target$clusterlcp <- cl_lcp
head(df_target)

# Plot all the sequences within each cluster para los 4 métodos
# OM1
seqIplot(seq_target, group = df_target$clusterom1, sortv = "from.start")
# OM2
# seqIplot(seq_target, group = df_target$clusterom2, sortv = "from.start")
# LCS
seqIplot(seq_target, group = df_target$clusterlcs, sortv = "from.start")
# LCP
seqIplot(seq_target, group = df_target$clusterlcp, sortv = "from.start")


######### Plot clusters' spatial distribution

# Elaborate raster OM1
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterom1))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_com1 <- raster(xyz)
plot(raster_com1)

# Elaborate raster OM2
# xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterom2))
# names(xyz) <- c("x","y","z")
# xyz <- xyz[complete.cases(xyz), ]
# coordinates(xyz) <- ~ x + y
# gridded(xyz) <- TRUE
# raster_com2 <- raster(xyz)
# plot(raster_com2)

# Elaborate raster LCS
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterlcs))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcs <- raster(xyz)
plot(raster_clcs)

# Elaborate raster LCP
xyz <- as.data.frame(cbind(df_target$x,df_target$y,df_target$clusterlcp))
names(xyz) <- c("x","y","z")
xyz <- xyz[complete.cases(xyz), ]
coordinates(xyz) <- ~ x + y
gridded(xyz) <- TRUE
raster_clcp <- raster(xyz)
plot(raster_clcp)

#######################################################################################
## Run discrepancy analyses to study how sequences are related to covariates

# Compute and test the share of discrepancy explained by different categories on covariates 
da1 <- dissassoc(dist.om1, group = tab$slopeK, R = 50)
print(da1$stat)
da2 <- dissassoc(dist.om1, group = tab$distK, R = 50)
print(da2$stat)
da3 <- dissassoc(dist.om1, group = tab$distK2, R = 50)
print(da3$stat)


# Selecting event subsequences:
# The analysis was restricted to sequences that exhibit the state Mosaic

tabe.seq <- seqecreate(tab.seq, use.labels = FALSE)
mosaic <- seqecontain(tabe.seq, event.list = c("Mosaic"))
mosaic_tab <- tab[mosaic,]
mosaic.seq <- tab.seq <- seqdef(mosaic_tab, 3:13, alphabet = alphabet, states = short_labels,
                                cpal = palette, labels = labels)
mosaic.seqe <- seqecreate(mosaic.seq, use.labels = FALSE)

# Look for frequent event subsequences and plot the 10 most frequent ones.
fsubseq <- seqefsub(mosaic.seqe, pmin.support = 0.05)
head(fsubseq)
# 10 Most common subsequences
plot(fsubseq[1:10], col = "grey98")

# Determine the subsequences of transitions which best discriminate the groups as
# areas close and faraway from roads
discr1 <- seqecmpgroup(fsubseq, group = mosaic_tab$distK2)
plot(discr1[1:10],cex=1,cex.legend=1,legend.title="Distance",cex.lab=0.8, cex.axis = 0.8)
# areas with moderate vs steep slope
discr2 <- seqecmpgroup(fsubseq, group = mosaic_tab$slopeK2)
plot(discr2[1:10],cex=1,cex.legend=1,legend.title="Slope",cex.lab=0.8, cex.axis = 0.8)
# clusters of sequences
discr3 <- seqecmpgroup(fsubseq, group = mosaic_tab$clusterom1)
plot(discr3[1:10],cex=1,cex.legend=1,legend.title="Clusters OM1",cex.lab=0.8, cex.axis = 0.8)

