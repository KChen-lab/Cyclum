# Two functions, read text matrices and binary matrices

# Note that in R, matrix is stored by columns. This issue is already resolved by these functions. No further action needed.

#' read_text: read matrix from text file
#' 
#' The file should start with n_row and n_col, followed by entries in the matrix
#' Row is contiguous
#' @param file_name: name of input file
read_text <- function(file_name){
  d <- scan(file_name)
  m <- matrix(d[-1:-2], nrow=d[2], ncol=d[1])
  return(t(m))
}

#' read_binary: read matrix from binary file
#' 
#' The file should start with n_row and n_col, in 4 byte (int32) little endian format
#' Then, the entires in the matrix in 8 byte (in python float, otherwise double) little endian format
#' @param file_name: name of input file
read_binary <- function(file_name){
  f <- file(file_name, 'rb')
  n_row <- readBin(f, what="int", n=1, size=4, endian="little")
  n_col <- readBin(f, what="int", n=1, size=4, endian="little")
  m2 <- matrix(nrow=n_col, ncol=n_row, 
               data=readBin(f, what="double", n=n_row*n_col, size=8, endian="little"))
  close(f)
  return(t(m2))
}

read_binary_with_name <- function(file_name_mask){
  m <- read_binary(paste(file_name_mask, "-value.bin", sep=""))
  all_names <- scan(paste(file_name_mask, "-name.txt", sep=""), what="character", sep = "\t")
  rownames(m) <- all_names[1:nrow(m)]
  colnames(m) <- all_names[-1:-nrow(m)]
  return(m)
}
