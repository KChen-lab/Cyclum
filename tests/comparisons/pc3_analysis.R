library(mclust)
library(Rtsne)
library(matrixStats)
setwd("~/PycharmProjects/SCAE")
source("R_codes/reader.R")

input_file_mask = "pc3"
pkm = t(read_binary_with_name(paste("data/McDavid/", input_file_mask, sep="")))
lab_file_name = paste("data/McDavid/", input_file_mask, "-label.txt", sep="")
lab = read.csv(lab_file_name, sep='\t', row.names=1)
emb <- read.csv(paste("results/McDavid/", input_file_mask, "-rec.csv", sep=""), sep='\t', row.names=1)
v <- read_binary(paste("results/McDavid/", input_file_mask, "-v.bin", sep=""))

groundtruth <- vector(length = dim(pkm)[2])
groundtruth[lab['stage', ] == 'g0/g1'] = 1
groundtruth[lab['stage', ] == 's'] = 2
groundtruth[lab['stage', ] == 'g2/m'] = 3

recat <- read.csv('results/McDavid/pc3_recat_emb.csv', row.names=1)

pdf('analysis_results/pc3_0.pdf')

scAE_embedding = emb$emb
mclust_result <- MclustDA(scAE_embedding, groundtruth)
plot(mclust_result, col=c('#FF0000', '#00FF00', '#0000FF'), what="scatterplot")
title(paste('scAE Accuracy:', round(1 - summary(mclust_result)$err, 3)))

pc = prcomp(t(pkm), center = FALSE, scale.=FALSE, rank. = 5)

mclust_result <- MclustDA(recat$x, groundtruth)
plot(mclust_result, col=c('#FF0000', '#00FF00', '#0000FF'), what="scatterplot")
title(paste('reCAT Accuracy:', round(1 - summary(mclust_result)$err, 3)))

PC1 <- pc$x[, 1]
mclust_result <- MclustDA(PC1, groundtruth)
plot(mclust_result, col=c('#FF0000', '#00FF00', '#0000FF'), what="scatterplot")
title(paste('PC1 Accuracy:', round(1 - summary(mclust_result)$err, 3)))

PC2 <- pc$x[, 2]
mclust_result <- MclustDA(PC2, groundtruth)
plot(mclust_result, col=c('#FF0000', '#00FF00', '#0000FF'), what="scatterplot")
title(paste('PC2 Accuracy:', round(1 - summary(mclust_result)$err, 3)))

plot.new()

legend("topleft", legend=c('G0/G1', 'S', 'G2/M'), col=c('#FF0000', '#00FF00', '#0000FF'), pt.bg=c('#FF000040', '#00FF0040', '#0000FF40'), pch=22)

