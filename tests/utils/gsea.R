# Help running gsea

mat2txt <- function(mat, filepath){
  #write.table(x = cbind(NAME=rownames(mat), DESCRIPTION='na', as.data.frame(mat)), 
  #          file = filepath, row.names=F, sep='\t', quote = F)
  data.table::fwrite(cbind(NAME=rownames(mat), DESCRIPTION='na', as.data.frame(mat)), 
                           file = filepath, row.names=F, sep='\t', quote = F)
}

vec2cls <- function(vec, name, filepath, continuous=TRUE){
  if(continuous){
    s = "#numeric\n"
    s = paste0(s, "#", name, '\n')
    s = paste0(s, paste(vec, collapse = ' '), '\n')
  } else {
    vec = as.character(vec)
    vec = gsub(" ", "", vec, fixed = TRUE)
    types = length(unique(vec))
    s = paste(length(vec), length(types), '1\n')
    s = paste0(s, '#', paste(types), '\n')
    s = paste0(s, paste(vec), '\n')
  }
  cat(s, file=filepath, append = F)
}

mat2cls <- function(mat, filepath, continuous=TRUE){
  if (continuous){
    s = "#numeric\n"
    phenotype_names = colnames(mat)
    for (i in 1:dim(mat)[2]){
      s = paste0(s, "#", phenotype_names[i], '\n')
      print(length(mat[, i]))
      s = paste0(s, paste(mat[, i], collapse = ' '), '\n')
    }
  } else {
    stop('GSEA does not support multiple categorical phenotypes in one file.')
  }
  cat(s, file=filepath, append = F)
}

dbfilter <- function(ifilepath, ofilepath, good_genes){
  s = ""
  for (filepath in ifilepath){
    msigdb <- read.table(filepath, header = F, col.names = F, sep='\t', stringsAsFactors = F)
    genes <- msigdb[c(-1, -2), ]
    print(sum(genes %in% good_genes))
    s = paste0(s, paste0(msigdb[c(T, T, genes %in% good_genes), ], collapse = '\t'), "\n")
  }
  cat(s, file=ofilepath)
}