# Wine-Quality-Score

The aim of this project is to train a model or ensemble of several models in order to predict wine quality score. The data consists of 11 features which are: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol; and 1 output variable which is quality of wine. There are two data sets for red and white wine types. Data sets will be concatenated, and general model is going to be trained for both red and white wine types.

Firstly, let’s observe data set structure and data types:

<p align="center">
  <img width="90%" height="90%" src="https://raw.githubusercontent.com/BatyaGG/Wine-Quality-Score/master/figures/structure.JPG">
  <br>
  <i>Table 1: Dataset structure</i>
</p>

All features are of numerical data types, no strings, factors and categories. So, no need to create dummy variables or to convert variables to numerical. No wine names or IDs, all features are chemical data and therefore cannot be deleted by first look. By checking for missing values, it was found that all observations are complete, so no need to impute data. Some features of data set may have constant or around constant value for all observations. Such features have no correlation to output variables and may lead to unstable models.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/nzv.JPG">
  <br>
  <i>Table 2: Features test on zeroVar and nzv</i>
</p>

Based on calculated frequency ratio and percent of unique values of features it is found that there are no features with zero or near-zero variances.

Data is already appropriate for regression analysis. To improve regression results and avoid feature dominance data should be standardized (mean = 0, variance = 1). plsRglm does standardization automatically. Also, red and white wine datasets are vertically concatenated and shuffled row-wise to distribute red and white wine observations. This is done to improve generalization of training for both red and white wine types.

# First proposed model
Let’s firstly try to fit plsRglm model on whole data set tuning two main parameters: number of components (nt) and level of significance for predictors (alpha.pvals.expli). Model tuning and fitting is done using 10-fold cross-validation test. From initial guess nt was ranging from 3 to 7 and alpha.pvals.expli ranged from 0.1 to 3. Best performance was 0.5697036 MAE at nt = 7 and alpha.pvals.expli = 1.550 to 3.0.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_parameters.png">
  <br>
  <i>Figure 1: RMSE vs plsRglm parameters relation</i>
</p>

Seems like lines on Figure 1 are of exponential nature. Line for 0.1 p-Value (red) is near to its asymptote at 7 #PLS components, however other lines are still have decreasing rate.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_parameters_2.png">
  <br>
  <i>Figure 2: RMSE vs plsRglm parameters relation</i>
</p>

Continuing cross-validation test it is found that nt = 9 and p-value ranging from 1 to 3 are parameters for most accurate model trained on raw dataset having cross-validated 0.56959 MAE. This is first proposed model.

# Outlier detection and elimination

Outliers in datasets may have impact on model accuracy to some extent. However, outlier deletion is not good approach in some cases. Wine dataset have many observations and not very highly dimensional, therefore outlier observations deletion may improve model accuracy or at least decrease training time. Let’s analyze dataset features distributions beforehand. Features have various metric units (mg/cm^3, g/dm^3 etc.). Also, they have different concentrations and therefore have varying ranges. So, normalizing dataset will make plots consistent to visualize.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/feature_distribution.png">
  <br>
  <i>Figure 3: Normalized feature distribution boxplot</i>
</p>

From Figure 3 it is seen that each feature of data frame has outliers. Testing is done using following procedure 10 times:

  1. Data was shuffled and splitted to training and testing sets by 3/1 ratio
  2. Training set was duplicated
  3. Observation in duplicate dataset having at least 1 outlier in any of its features was eliminated
  4. Two plsRglm models (with default parameters) were trained each for raw and clean training sets
  5. Both models were tested on same testing datasets
  6. Mean average errors were saved and compared

Outlier detection was done using univariate approach with a help of boxplot.stats() function. Values considered as outliers if they lie outside 1.5 * “Inter Quartile Range” (IQR) where IQR is difference between 75th and 25th quartiles.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/raw_vs_clean.JPG">
  <br>
  <i>Table 3: Comparison of model accuracies for raw vs clean datasets</i>
</p>

Table 3 shows that both models trained on raw and clean datasets have in general similar performance. Comparing their MAEs 10 times both models have equal number of wins (5 times each). However, clear data have less observations by 1000 rows in average. It is about 20% decrease of size of dataset leading to faster training. In case of model stacking (ensembling), using clean dataset could be useful in terms of training time.

# Training separate models for red and white wine types

Red and white wine datasets have differing feature values in terms of range and mean values. According to Cortez et al. red and white datasets have following properties.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/feature_props.png">
  <br>
  <i>Table 4: Red and white wine feature properties</i>
</p>

Since datasets have different properties, individual models for red and white wine types probably have better accuracies than general model.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/red_param_tuning.png">
  <br>
  <i>Figure 4: Red wine plsRglm model performance for different model parameters</i>
</p>

From Figure 4 it is seen that at parameters nt = (4 – 10) and p-value = 0.1 model have best accuracy having 0.5039640 MAE.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/white_param_tuning.png">
  <br>
  <i>Figure 5: White wine plsRglm model performance for different model parameters</i>
</p>

From Figure 5 it is seen that best parameters for white wine model is nt = 6 and p = 0.6 and at such parameters accuracy is 0.5862921 MAE. Extending CV analysis better results were achieved at nt = 6 and p = 0.1 with 0.5861152 MAE. Training on clean white wine data slightly improved accuracy to 0.5824022.

Next challenge is finding a way to distinguish wine types before training and predicting. One possible way is checking properties of observation features for satisfying Table 4. Each wine type feature has 3 properties (min, max and mean) and each new observation feature will be checked for satisfying those properties. For each feature a satisfaction score will be calculated for each wine type. Results from each feature will be considered and exceeding wine type will be chosen. Such function was implemented and tested. As a result, red dataset observations were correctly classified in 99.1% and white wine observations were correctly classified in 87.2% occurrences. Best features for red/white wine classification were found empirically (all except citric.acid and alcohol) and finally red and white wines were correctly classified in 97% for both. Now we can train individual models for red and white wine and combine them in ensemble.

# Feature elimination attempt
Let’s try to decrease complexity of datasets by feature elimination. Ranking features by their relevance is a first step of feature selection process. Recursive feature elimination algorithm is used for feature selection process. A random forest algorithm is trained for each 10-fold cross-validation iteration on different feature subsets.

<p align="center">
  <img width="90%" height="90%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_feature.png">
  <br>
  <i>Figure 6: RMSE vs feature # graph</i>
</p>

From Figure 6 it is seen that all features have influence on accuracy at different degrees. Ranking of features is as follows:
1) Alcohol
2) Volatile Acidity
3) Free Sulfur Dioxide
4) Sulphates
5) pH
6) Residual Sugar
7) Citric Acid
8) Total Sulfur Dioxide
9) Fixed Acidity
10) Chlorides
11) Density Deleting least features could be a choice to decrease complexity of training, however there will be decrease in accuracy. There is no any obviously useless feature, and since the aim of this part is improving accuracy, no feature will be deleted.
