# Function takes data as input and returns data subset
# without outlier datapoints

clear_data <- function(data){
  
  # each feature of input data is analysed
  for (i in 1:ncol(data)){
    # particular feature observations
    vec <- data[, i]
    
    # values those are out of 1.5 * IQR
    vec_out <- boxplot.stats(vec)$out
    
    # all outlier values found in feature vector assigned as NA
    vec[vec %in% vec_out] <- NA
    
    # data feature is updated
    data[, i] <- vec
  }
  
  # only complete observation data subset is returned
  data[complete.cases(data), ]
}