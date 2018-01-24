wine_type <- function(data){
  propertie_mat <- matrix( 
    c(4.6, 15.9, 8.3, 3.8, 14.2, 6.9,
      0.1, 1.6, 0.5, 0.1, 1.1, 0.3,
      0.0, 1.0, 0.3, 0.0, 1.7, 0.3,
      0.9, 15.5, 2.5, 0.6, 65.8, 6.4,
      0.01, 0.61, 0.08, 0.01, 0.35, 0.05,
      1, 72, 14, 2, 289, 35,
      6, 289, 46, 9, 440, 138,
      0.99, 1.004, 0.996, 0.987, 1.039, 0.994,
      2.7, 4, 3.3, 2.7, 3.8, 3.1,
      0.3, 2, 0.7, 0.2, 1.1, 0.5,
      8.4, 14.9, 10.4, 8, 14.2, 10.4),
    nrow=11, 
    ncol=6,
    byrow = TRUE)
  type_vector <- c()
  feat_vec <- c(1, 2, 4, 5, 6, 7, 8, 9, 10)
  for (i in 1:nrow(data)){
    obs <- data[i,]
    score_red <- 0
    for (j in feat_vec){
      feature_vector <- propertie_mat[j, ] - rep(obs[[j]], 6)
      r <- 0
      w <- 0
      if (feature_vector[1] <= 0){
        r = r + 1
      }
      if (feature_vector[2] >= 0){
        r = r + 1
      }
      if (abs(feature_vector[3]) <= abs(feature_vector[6])){
        r = r + 1
      }else{
        w = w + 1
      }
      if (feature_vector[4] <= 0){
        w = w + 1
      }
      if (feature_vector[5] <= 0){
        w = w + 1
      }
      if(r > w){
        score_red <- score_red + 1
      }
    }
    if (score_red > as.integer(length(feat_vec)/2)){
      type_vector <- c(type_vector, TRUE)
    }else{
      type_vector <- c(type_vector, FALSE)
    }
  }
  return(type_vector)
}