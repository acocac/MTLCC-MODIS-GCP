require(data.table)
require(raster)
require(maptools)
require(rgdal)
require(sf)

in_dir <- paste0("E:/acocac/research/AMZ/verification/sampling/raw")
out_dir <- paste0("E:/acocac/research/AMZ/verification/sampling/final")
dir.create(out_dir, showWarnings = FALSE, recursive = T)

tiles <- c('tile_0_201','tile_0_143', 'tile_1_463', 'tile_1_438', 'tile_0_630', 'tile_1_713','tile_0_365')
#tiles <- c('tile_0_201','tile_0_201')
#datasets <- c('MCD12Q1v6raw_LCType1','Copernicusnew_cf2others','mapbiomas')
datasets <- c('MCD12Q1v6stable_LCProp2')
ratio <- 0.01
splits <- c('stable')

for (d in datasets){
  
  if (d %in% c('mapbiomas','MCD12Q1v6stable_LCProp2')){
    schemes <- 'raw' #raw
  } else {
    schemes <- c('raw','simple') #simple
  }
  
  N <- length(tiles)
  spl_pol <- vector("list", N)
  spl_point <- vector("list", N)
  
  for (scheme in schemes){
    
    for (split in splits){
      
      for (i in seq(1,length(tiles))){
        
        fileName.sampling <- list.files(in_dir, pattern = paste0("*",split,"_",d,"_",scheme,"_",tiles[i]), full.names = T) #list patch files 
        print(fileName.sampling)
        sampling.temp <- rbindlist(lapply(fileName.sampling, fread, header = TRUE, sep = ",", na.strings=c("NA", '')))

        #point
        sampling.temp.sp <- sampling.temp
        coordinates(sampling.temp.sp) <- ~x + y
        proj4string(sampling.temp.sp) <- CRS("+init=epsg:4326")
        sampling.temp.sp$PLOTID2 <- paste0(tiles[i],"_",rownames(as.data.frame(sampling.temp.sp)))
        sampling.temp.sp$tile <- tiles[i]
        sampling.temp.sp$dyear <-tail(strsplit(fileName.sampling,"_|.csv")[[1]],1)
        
        spl_point[[i]] <- sampling.temp.sp

        #polygon
        raster_file <- paste0('E:/acocac/research/',tiles[i],'/eval/verification/watermask/2018/watermask/0_0_0.tif')
        r <- raster(raster_file)

        cells <- raster::cellFromXY(r, sampling.temp.sp)

        r[] <- NA
        r[cells] <- 1

        pol <- rasterToPolygons(r)

        # join objects
        pol.sf <- st_as_sf(pol)
        point.sf <- st_as_sf(sampling.temp.sp)

        pol_srdf <- st_join(pol.sf, point.sf)

        pol_srdf <- as(pol_srdf, "Spatial")

        spl_pol[[i]] <- pol_srdf
      }

      final_point <- do.call(bind, spl_point)
      final_pol <- do.call(bind, spl_pol)

      final_point$PLOTID <- rownames(as.data.frame(final_point))
      final_point$SAMPLEID <- final_point$PLOTID

      final_pol$PLOTID <- rownames(as.data.frame(final_pol))

      print(paste0('Dataset ',d,' & Scheme ',scheme))
      print(paste0('split = ',split))
      print(table(final_pol$tile))
      print(paste0('size = ',dim(final_pol)[1]))

      writeOGR(final_point, dsn=out_dir, layer=paste0(d,'_',scheme,'_',split,'_point'), driver = 'ESRI Shapefile', overwrite_layer = T)

      writeOGR(final_pol, dsn=out_dir, layer=paste0(d,'_',scheme,'_',split,'_pol'), driver = 'ESRI Shapefile', overwrite_layer = T)
    }
  }
}




