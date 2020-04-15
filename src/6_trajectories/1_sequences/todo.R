
#yearls <- paste0("c",as.character(seq(lc_target_years[1],lc_target_years[2],1)))
yearls <- paste0(as.character(seq(lc_target_years[1],lc_target_years[2],1)))

tab.seq <- seqdef(tab, 3:(length(yearls)+2), alphabet = alphabet, states = short_labels,
                  cpal = palette, labels = short_labels)

## Adjust some graphical parameters.
par( # change the margins
  lwd = 0.1, # increase the line thickness
  cex.axis = 0.5 # increase default axis label size
)

par() 
## Draw boxplot with no axes.

seqfplot(tab.seq, with.legend = F, cex.axis=1.2, main=paste0("Most common sequences for det in ",paste(terrai_target_years,collapse=" to "),"\n N=",dim(tab)[1]),  xaxt="n", xlab="")

## Draw x-axis without labels.
axis(side = 1, labels = FALSE)

## Draw the x-axis labels.
text(x = 1:length(yearls),
     ## Use names from the data list.
     labels = yearls,
     ## Change the clipping region.
     xpd = NA,
     ## Rotate the labels by 35 degrees.
     srt = 45,
     ## Adjust the labels to almost 100% right-justified.
     adj = 0.5,
     ## Increase label size.
     cex = 1.2)