# hdf read write

library(hdf5r)
mat2hdf <- function(mat, filepath){
  f <- H5File$new(filepath, mode = "w")
  f[['matrix']] <- mat
  if(!is.null(colnames(mat))) f[['colnames']] <- colnames(mat)
  if(!is.null(rownames(mat))) f[['rownames']] <- rownames(mat)
  h5close(f)
}

hdf2mat <- function(filepath){
  f <- H5File$new(filepath, mode = "r")
  if (f$exists('matrix')){
    mat <- f[['matrix']]$read() 
  } else {
    h5close(f)
    stop("no matrix found.")
  }
  if (f$exists('colnames')) colnames(mat) <- f[['colnames']]$read()
  if (f$exists('rownames')) rownames(mat) <- f[['rownames']]$read()
  h5close(f)
  return(mat)
}