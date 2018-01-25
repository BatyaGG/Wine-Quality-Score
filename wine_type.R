# Function takes data as input and returns red_wine
# logical vector having TRUE for red wine datapoints
# and FALSE for white wine datapoints

wine_type <- function(data){
  
  # Feature properties matrix as in Table 4 in report.
  # Each row corresponds to feature and first three
  # columns corresponds to min max and mean values for
  # red wine dataset. Fourth, fifth and sixth columns are
  # also for min max and mean values for white wine dataset.
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
  
  # initialization red_wine logical vector 
  # and features to be analyzed
  type_vector <- c()
  feat_vec <- c(1, 2, 4, 5, 6, 7, 8, 9, 10)
  
  # for each observation in input data a score is given
  for (i in 1:nrow(data)){
    obs <- data[i,]
    score_red <- 0
    
    # from feature vector in propertie matrix, observation 
    # feature value is substracted
    for (j in feat_vec){
      feature_vector <- propertie_mat[j, ] - rep(obs[[j]], 6)
      r <- 0
      w <- 0
      
      # if observation feature value is more than that feature min value
      # in propertie matrix, then probability of redness of datapoint is bigger
      if (feature_vector[1] <= 0){
        r = r + 1
      }
      
      # if observation feature value is less than that feature max value
      # in propertie matrix, then probability of redness of datapoint is bigger
      if (feature_vector[2] >= 0){
        r = r + 1
      }
      
      # if observation feature value is closer to that feature mean value
      # in propertie matrix, then probability of redness of datapoint is bigger,
      # else, probability of whiteness of datapoint is bigger
      if (abs(feature_vector[3]) <= abs(feature_vector[6])){
        r = r + 1
      }else{
        w = w + 1
      }
      
      # same logic as for red wine
      if (feature_vector[4] <= 0){
        w = w + 1
      }
      
      # same logic as for red wine
      if (feature_vector[5] <= 0){
        w = w + 1
      }
      
      # if observation feature satisfies to propertie matrix for
      # particular red wine feature more than white, then red wine
      # score is increased
      if(r > w){
        score_red <- score_red + 1
      }
    }
    
    # after all features are considered, if score_red is more than
    # half of the number of features considered, then current
    # observation is red wine datapoint, else, it is white wine datapoint
    if (score_red > as.integer(length(feat_vec)/2)){
      type_vector <- c(type_vector, TRUE)
    }else{
      type_vector <- c(type_vector, FALSE)
    }
  }
  
  # returning red_wine logical vector
  type_vector
}