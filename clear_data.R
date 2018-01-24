clear_data <- function(data){
  cols <- 1:ncol(data)
  # cols <- cols[-except]
  for (i in cols){
    vec <- data[, i]
    vec_out <- boxplot.stats(vec)$out
    vec[vec %in% vec_out] = NA
    data[, i] <- vec
  }
  return(data[complete.cases(data), ])
}