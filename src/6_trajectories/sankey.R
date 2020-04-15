landuse <- list(
  nodes = data.frame(
    name = c(
      NA, NA, "1.1.1. Formação Florestal", "1.1.2. Formação Savanica", NA, NA,
      NA, NA, NA, NA, NA, NA, NA, NA, "3.1. Pastagem", NA, NA, NA, 
      "3.2.1. Cultura Anual e Perene", NA, 
      "3.3. Mosaico de Agricultura e Pastagem", NA, NA, 
      "4.2. Infraestrutura Urbana", "4.5. Outra Área não Vegetada", NA, NA, NA,
      NA, NA, NA, NA,"5.1 Rio ou Lago ou Oceano"
    ),
    stringsAsFactors = FALSE
  ),
  links = data.frame(
    source = c(
      3L, 3L, 3L, 4L, 4L, 4L, 4L, 4L, 4L, 15L, 15L, 15L, 15L, 15L, 15L, 15L, 
      19L, 19L, 19L, 19L, 21L, 21L, 21L, 21L, 21L, 21L, 24L, 25L, 25L, 25L, 33L
    ),
    target = c(
      3L, 21L, 4L, 21L, 15L, 3L, 25L, 4L, 33L, 19L, 15L, 21L, 3L, 25L, 4L, 33L,
      15L, 19L, 4L, 21L, 4L, 21L, 25L, 33L, 15L, 3L, 4L, 25L, 4L, 33L,33L
    ),
    value = c(
      0.544859347827813, 0.00354385993588971, 0.494359662221154, 
      4.67602736159475, 2.20248911690968, 0.501437742068369,
      0.00354375594818463, 24.8427814053755, 0.439418727642527,
      0.0079740332093807, 11.8060486886398, 2.76329829691466,
      0.000886029792298199, 0.00177186270758855, 3.35504921147758,
      0.14263144351167, 1.12170804870686, 0.0478454594554582,
      0.217079959877658, 0.00620223918980076, 1.79754946594068,
      9.02868098124075, 0.00442981113709027, 0.242743895018645,
      0.498770814980772, 0.00265782877794886, 0.000885894856554407,
      0.379188333632346, 0.00265793188317263, 0.00265771537700804,
      0.39158027235054
    ),
    stringsAsFactors = FALSE
  )
)

# create a links data frame where the right and left column versions of each node
# are distinguishble
links <- 
  data.frame(source = paste0(landuse$nodes$name[landuse$links$source], " (1985)"),
             target = paste0(landuse$nodes$name[landuse$links$target], " (2017)"),
             value = landuse$links$value,
             stringsAsFactors = FALSE)

# build a nodes data frame from the new links data frame
nodes <- data.frame(name = unique(c(links$source, links$target)), 
                    stringsAsFactors = FALSE)

# change the source and target variables to be the zero-indexed position of
# each node in the new nodes data frame
links$source <- match(links$source, nodes$name) - 1
links$target <- match(links$target, nodes$name) - 1

# remove the year indicator from the node names
nodes$name <- substring(nodes$name, 1, nchar(nodes$name) - 7)

# plot it
library(networkD3)
sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
              Target = "target", Value = "value", NodeID = "name",
              units = "km²", fontSize = 12, nodeWidth = 30)