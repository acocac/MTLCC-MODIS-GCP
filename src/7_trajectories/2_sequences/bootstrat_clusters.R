############################## Section 1: sequence analysis
############################
############################### install necessary packages
#############################
##source
###http://www.llcsjournal.org/index.php/llcs/article/view/409/514

# TraMineR is a R-package for describing, summarizing, analyzing and rendering discrete
install.packages(TraMineR)
require(TraMineR) # load the TraMineR package for sequence analysis
# The WeightedCluster library provides functions to cluster states sequences and weighted
data. install.packages(WeightedCluster)
require(WeightedCluster) # load the WeightedCluster package for cluster quality analysis
# fpc package provides bootstrapping methods and statistics for clustering.
install.packages(fpc)
require(fpc) # load the fpc package for bootstrapping
################################ set working directory
###############################
setwd ("replace this with your working directory")
################################## load sequence data
##############################
load("data_monthly_sequence.RData") # replace RData with your own sequence data
#################### here we start preparing the analysis with TraMineR
###################
data.lab <- c("Married", "Married with Children", "Single", "Single with Children",
              "Union", "Union with Children") # 6 demographic events
data.alph <- c("M","MC","S","SC","U","UC") # define the sequence alphabet with these 6
events
data.shortlab <- c("M","MC","S","SC","U","UC") # abbreviation of these 6 demographic
events
data.seq <- seqdef(data_monthly_sequence, states = data.shortlab, labels = data.lab,
                   alphabet = data.alph) # creates a state sequence object with attributes such as alphabet
and state labels
######################### calculate sequence distance matrix c4.dis
######################
############################# using optimal matching (OM)
############################
####### insertion and deletion cost equals 4, substitution-cost is constant (default value =
2) ######
data.dis <- seqdist(data.seq, indel = 4,method = "OM",sm = "CONSTANT")
############################# other option see help file seqdist
#########################
############# here we apply hierarchical cluster analysis using Ward’s method
#############
hc.ward <- hclust(as.dist(data.dis), method = "ward.D")
######### here we compare cluster quality between 2 and 8 clusters ###########
mvad <- wcKMedRange(data.dis, kvals = 2:8,initialclust = hc.ward)
######### here we apply bootstrapping for 7 clusters as an example ###########
data.hc <- clusterboot(data.dis, distances = TRUE, clustermethod = disthclustCBI, method =
                         "ward.D", k = 7)
### here we perform data visualization for the cluster solution to gain substantive
understanding ##
###################### We take 7 clusters as an example #####################
#################### and require sequence index plots #####################
seqIplot(data.seq, group = data.hc$partition, border = NA, weighted = FALSE, sortv =
           "from.start")
#################### sequence medoid plot for cluster number equals 7
###################
################## calculate sequence medoid for cluster number equals 7
#################
icenter <- disscenter(data.dis, factor(data.hc$partition), medoids.index="first")
seqiplot(data.seq[icenter,]) ## plot calculated medoid for the 7-cluster solution
######################### Section 2: latent class analysis
############################
############################### install necessary packages
#############################
# poLCA is a R package for latent class analysis and latent class regression models for
polytomous #outcome variables. Also known as Latent Structure Analysis.
install.packages(poLCA)
require(poLCA) # load the poLCA package for latent class analysis
################# we start preparing the latent class analysis using poLCA
###################
#################### We use the same dataset as in sequence analysis
###################
#################### rename the monthly sequence states chronologically
################
names(data_monthly_sequence)[8:151] <- paste("m", 1:144, sep = "")
#################### define variables in latent class analysis ###################
var <- cbind(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
             m11, m12, m13, m14, m15, m16, m17, m18, m19, m20,
             m21, m22, m23, m24, m25, m26, m27, m28, m29, m30,
             m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
             m41, m42, m43, m44, m45, m46, m47, m48, m49, m50,
             m51, m52, m53, m54, m55, m56, m57, m58, m59, m60,
             m61, m62, m63, m64, m65, m66, m67, m68, m69, m70,
             m71, m72, m73, m74, m75, m76, m77, m78, m79, m80,
             m81, m82, m83, m84, m85, m86, m87, m88, m89, m90,
             m91, m92, m93, m94, m95, m96, m97, m98, m99, m100,
             m101, m102, m103, m104, m105, m106, m107, m108, m109, m110,
             m111, m112, m113, m114, m115, m116, m117, m118, m119, m120,
             m121, m122, m123, m124, m125, m126, m127, m128, m129, m130,
             m131, m132, m133, m134, m135, m136, m137, m138, m139, m140,
             m141, m142, m143, m144)~1
#################### here we perform latent class analysis ###################
#################### with 7 latent classes as an example ###################
## the number of random starting values nrep may be changed
m7 <- poLCA(var, data_monthly_sequence, nclass = 7, nrep = 500)
### Getting model fit statistic BIC and relative entropy for latent between 4 and 8.
#################### BIC calculated in a for loop ###################
BIC <- NULL
for (i in 4: 8)
{
  m <- poLCA(var, data_monthly_sequence, nclass = i, nrep = 500)
  BIC[i] <- m$bic
}
#################### relative entropy calculated in a for loop ###################
entropy <- NULL
for (i in 4: 8)
{
  m <- poLCA(var, data_sts, nclass = i, nrep = 100)
  entropy[i] <- poLCA.entropy(m)
}
# here we perform data visualization for the classification solution to gain substantive
understanding
###### data visualization of latent class analysis classification using TraMineR package
#########
###################### and use the 7-cluster solution as an example
#####################
class=m7$predclass
#################### sequence index plot for class number equals 7
#####################
seqIplot(data.seq, group class, border = NA, weighted = FALSE, sortv = "from.start")
#################### sequence model plot for cluster number equals 7
###################
seqmsplot(data.seq, group = class, border = NA, weighted = FALSE)
######################### Section 3: Typology Comparison
############################
############################### install necessary packages
#############################
# Software for multinomial log-linear models.
install.packages(nnet)
require(nnet) # load the nnet package for multinomial logistic regression analysis
#################### load data of cluster and classification solution
#####################
#################### load data of background variable #####################
load("background.Rda")
#################### load sequence analysis results #####################
load("Sequence_solution.RData")
#################### load latent class analysis results #####################
load("Latent_class_solution.RData")
#################### factorize all categorical variable #####################
f_Sequence_solution <- factor(Sequence_solution)
f_ Latent_class_solution <- factor(Latent_class_solution)
f_background_country <- factor(background$country)
#################### using Netherland as reference country #####################
f_background_country <- relevel(f_background_country, ref = "13")
#################### using traditional as reference cluster #####################
f_Sequence_solution <- relevel f_Sequence_solution, ref = "3")
#################### using traditional as reference cluster #####################
############# perform multinomial logistic regression on cluster solution ##########
glm.fit.sa=multinom(country ~ f_Sequence_solution, data= f_Sequence_solution)
################################### calculate BIC
####################################
BIC(glm.fit.sa)
######################### Predict probability ##################################
dses <- data.frame(c("1", "2", "3", "4", "5", "6")) ###create a 6 –cluster dataset for
sequence analysis
names(dses) <- "sequence_solution"
predict(glm.fit.sa, newdata = dses, "probs")