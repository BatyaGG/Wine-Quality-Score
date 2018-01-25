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
