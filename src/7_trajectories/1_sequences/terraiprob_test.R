pr

dir <- 'F:/acoca/research/gee/dataset/tile1/terrai/detection_2004_01_01_to_2014_12_19'
pattern <- '2012_'
  
files <- list.files(path=dir, pattern="*.asc$", all.files=FALSE, full.names=TRUE,recursive=TRUE)

files <- Filter(function(x) grepl(pattern, x), files)

probserie <- raster::stack(files)

prob_mean<- calc(probserie, mean)
prob_max<- calc(probserie, max)
prob_min<- calc(probserie, min)

boxplot(probserie, xlab="Cluster", ylab="Elevation")


plot(prob_mea
     n)
