# Wine-Quality-Score

The aim of this project is to train a plsRglm model or ensemble of several plsRglm models in order to predict wine quality score. The data consists of 11 features which are: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol; and 1 output variable which is quality of wine. There are two data sets for red and white wine types. Data sets will be concatenated, and general model is going to be trained for both red and white wine types.

## Sources

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Installation

Clone or download the project

Following packages are necessary to run script: caret <6.0-77>, plsRglm <1.1.1>

External packages for visualizations used in report: ggplot2 <2.2.1>, psych <1.7.8>

## Usage

Import caret package and evaluate directory of script folder:
```
library(caret)
script_dir = dirname(sys.frame(1)$ofile)
```

Create custom caret model using list datastructure:
```
custom_model <- list(library = "plsRglm",
                     loop = NULL,
                     type = "Regression",
                     parameters = data.frame(parameter = c("nt.red", "alpha.pvals.expli.red",
                                                           "nt.white", "alpha.pvals.expli.white",
                                                           "modele.red", "modele.white", "pca.prep"),
                                             class = c(rep("numeric", 4), rep("character", 2),
                                                       "logical"),
                                             label = c("#PLS Components (red)", "p-Value threshold (red)",
                                                       "#PLS Components (white)", "p-Value threshold (white)",
                                                       "PLS Model (red)", "PLS Model (white)",
                                                       "PCA preprocessed")),
                     grid = function(x, y, len = NULL, search = "grid") {
                       out <-  expand.grid(nt.red = 4, 
                                           alpha.pvals.expli.red = 0.13,
                                           nt.white = 6,
                                           alpha.pvals.expli.white = 0.1,
                                           modele.red = "pls-glm-poisson",
                                           modele.white = "pls-glm-inverse.gaussian",
                                           pca.prep = FALSE)
                       out
                     },
                     fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                       source(paste(script_dir, "clear_data.R", sep = "/"))
                       source(paste(script_dir, "wine_type.R", sep = "/"))
                       is_red <- wine_type(x)
                       data <- data.frame(x, y)
                       data_red <- data[is_red,]
                       data_white <- data[!is_red,]
                       data_red <- clear_data(data_red)
                       data_white <- clear_data(data_white)
                       x_red <- data_red[,-(ncol(x) + 1)]
                       y_red <- data_red[,(ncol(x) + 1)]
                       x_white <- data_white[,-(ncol(x) + 1)]
                       y_white <- data_white[,(ncol(x) + 1)]
                       if (param$pca.prep){
                         trans_red <- preProcess(x_red, method = "pca")
                         trans_white <- preProcess(x_white, method = "pca")
                         x_red <- predict(trans_red, newdata = x_red)
                         x_white <- predict(trans_white, newdata = x_white)
                       }
                       capture.output(mod_red <- plsRglm::plsRglm(y_red, x_red,
                                                 nt = param$nt.red,
                                                 modele = param$modele.red,
                                                 pvals.expli = param$alpha.pvals.expli.red < 1,
                                                 sparse  = param$alpha.pvals.expli.red < 1,
                                                 alpha.pvals.expli = param$alpha.pvals.expli.red,
                                                 ...))
                       capture.output(mod_white <- plsRglm::plsRglm(y_white, x_white,
                                                   nt = param$nt.white,
                                                   modele = param$modele.white,
                                                   pvals.expli = param$alpha.pvals.expli.white < 1,
                                                   sparse  = param$alpha.pvals.expli.white < 1,
                                                   alpha.pvals.expli = param$alpha.pvals.expli.white,
                                                   ...))
                       mod <- list(pca_prep=param$pca.prep, red=mod_red, white=mod_white)
                       
                       if (param$pca.prep){
                         mod <- list(pca_prep=param$pca.prep, red=mod_red, white=mod_white,
                                     red_trans=trans_red, white_trans=trans_white)
                       }
                       mod
                     },
                     predict = function(modelFit, newdata, submodels = NULL){
                       newdata$order <- 1:nrow(newdata)
                       is_red <- wine_type(newdata)
                       data_red <- newdata[is_red,]
                       data_white <- newdata[!is_red,]
                       if (modelFit[[1]]){
                         red_order <- data_red$order
                         white_order <- data_white$order
                         data_red <- predict(modelFit[[4]], newdata = data_red[, -ncol(data_red)])
                         data_white <- predict(modelFit[[5]], newdata = data_white[, -ncol(data_white)])
                         data_red$order <- red_order
                         data_white$order <- white_order
                       }
                       out_red <- predict(modelFit[[2]], data_red[, -ncol(data_red)], type="response")
                       out_white <- predict(modelFit[[3]], data_white[, -ncol(data_red)], type="response")
                       out <- data.frame(out = c(out_red, out_white))
                       out$order <- c(data_red$order, data_white$order)
                       out <- out[order(out$order),]
                       round(out$out)
                     },
                     prob = NULL,
                     sort = function(x) {
                       x[order(-x$alpha.pvals.expli.red, x$nt.red,
                               -x$alpha.pvals.expli.white, x$nt.white),]
                       }
                     )
```

Read red and white wine csv files and concatenate them to one variable:
```
data_red = read.csv(paste(script_dir, "winequality-red.csv", sep = "/"), header = TRUE, sep = ";")
data_white = read.csv(paste(script_dir, "winequality-white.csv", sep = "/"), header = TRUE, sep = ";")
data <- rbind(data_red, data_white)
```

Split your data to training and testing datasets with desired ratio (80% training and 20% testing in this case):
```
index <- createDataPartition(data$quality, p=0.8, list=FALSE)
train_set <- data[index,]
test_set <- data[-index,]
```

Train your model:
```
model <- train(y = data[, 12], x = data[, -12], method = custom_model)
```

Test your model by predicting test dataset outputs and evaluating accuracy using caret::postResample():
```
predictions <- predict(model_or, newdata=test_set_proc[, -10])
postResample(pred = predictions, obs = test_set[, "quality"])
```

# Project report

Firstly, let’s observe data set structure and data types:

<p align="center">
  <img width="80%" height="80%" src="https://raw.githubusercontent.com/BatyaGG/Wine-Quality-Score/master/figures/structure.JPG">
  <br>
  <i>Table 1: Dataset structure</i>
</p>

All features are of numerical data types, no strings, factors and categories. So, no need to create dummy variables or to convert variables to numerical. No wine names or IDs, all features are chemical data and therefore cannot be deleted by first look. By checking for missing values, it was found that all observations are complete, so no need to impute data. Some features of data set may have constant or around constant value for all observations. Such features have no correlation to output variables and may lead to unstable models.

<p align="center">
  <img width="70%" height="70%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/nzv.JPG">
  <br>
  <i>Table 2: Features test on zeroVar and nzv</i>
</p>

Based on calculated frequency ratio and percent of unique values of features it is found that there are no features with zero or near-zero variances.

Data is already appropriate for regression analysis. To improve regression results and avoid feature dominance data should be standardized (mean = 0, variance = 1). plsRglm does standardization automatically. Also, red and white wine datasets are vertically concatenated and shuffled row-wise to distribute red and white wine observations. This is done to improve generalization of training for both red and white wine types.

## First proposed model
Let’s firstly try to fit plsRglm model on whole data set tuning two main parameters: number of components (nt) and level of significance for predictors (alpha.pvals.expli). Model tuning and fitting is done using 10-fold cross-validation test. From initial guess nt was ranging from 3 to 7 and alpha.pvals.expli ranged from 0.1 to 3. Best performance was 0.5697036 MAE at nt = 7 and alpha.pvals.expli = 1.550 to 3.0.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_parameters.png">
  <br>
  <i>Figure 1: RMSE vs plsRglm parameters relation</i>
</p>

Seems like lines on Figure 1 are of exponential nature. Line for 0.1 p-Value (red) is near to its asymptote at 7 #PLS components, however other lines are still have decreasing rate.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_parameters_2.png">
  <br>
  <i>Figure 2: RMSE vs plsRglm parameters relation</i>
</p>

Continuing cross-validation test it is found that nt = 9 and p-value ranging from 1 to 3 are parameters for most accurate model trained on raw dataset having cross-validated 0.56959 MAE. This is first proposed model.

## Outlier detection and elimination

Outliers in datasets may have impact on model accuracy to some extent. However, outlier deletion is not good approach in some cases. Wine dataset have many observations and not very highly dimensional, therefore outlier observations deletion may improve model accuracy or at least decrease training time. Let’s analyze dataset features distributions beforehand. Features have various metric units (mg/cm^3, g/dm^3 etc.). Also, they have different concentrations and therefore have varying ranges. So, normalizing dataset will make plots consistent to visualize.

<p align="center">
  <img width="70%" height="70%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/feature_distribution.png">
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
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/raw_vs_clean.JPG">
  <br>
  <i>Table 3: Comparison of model accuracies for raw vs clean datasets</i>
</p>

Table 3 shows that both models trained on raw and clean datasets have in general similar performance. Comparing their MAEs 10 times both models have equal number of wins (5 times each). However, clear data have less observations by 1000 rows in average. It is about 20% decrease of size of dataset leading to faster training. In case of model stacking (ensembling), using clean dataset could be useful in terms of training time.

## Training separate models for red and white wine types

Red and white wine datasets have differing feature values in terms of range and mean values. According to Cortez et al. red and white datasets have following properties.

<p align="center">
  <img width="70%" height="70%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/feature_props.png">
  <br>
  <i>Table 4: Red and white wine feature properties</i>
</p>

Since datasets have different properties, individual models for red and white wine types probably have better accuracies than general model.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/red_param_tuning.png">
  <br>
  <i>Figure 4: Red wine plsRglm model performance for different model parameters</i>
</p>

From Figure 4 it is seen that at parameters nt = (4 – 10) and p-value = 0.1 model have best accuracy having 0.5039640 MAE.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/white_param_tuning.png">
  <br>
  <i>Figure 5: White wine plsRglm model performance for different model parameters</i>
</p>

From Figure 5 it is seen that best parameters for white wine model is nt = 6 and p = 0.6 and at such parameters accuracy is 0.5862921 MAE. Extending CV analysis better results were achieved at nt = 6 and p = 0.1 with 0.5861152 MAE. Training on clean white wine data slightly improved accuracy to 0.5824022.

Next challenge is finding a way to distinguish wine types before training and predicting. One possible way is checking properties of observation features for satisfying Table 4. Each wine type feature has 3 properties (min, max and mean) and each new observation feature will be checked for satisfying those properties. For each feature a satisfaction score will be calculated for each wine type. Results from each feature will be considered and exceeding wine type will be chosen. Such function was implemented and tested. As a result, red dataset observations were correctly classified in 99.1% and white wine observations were correctly classified in 87.2% occurrences. Best features for red/white wine classification were found empirically (all except citric.acid and alcohol) and finally red and white wines were correctly classified in 97% for both. Now we can train individual models for red and white wine and combine them in ensemble.

## Feature elimination attempt
Let’s try to decrease complexity of datasets by feature elimination. Ranking features by their relevance is a first step of feature selection process. Recursive feature elimination algorithm is used for feature selection process. A random forest algorithm is trained for each 10-fold cross-validation iteration on different feature subsets.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/RMSE_vs_feature.png">
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
11) Density

Deleting least features could be a choice to decrease complexity of training, however there will be decrease in accuracy. There is no any obviously useless feature, and since the aim of this part is improving accuracy, no feature will be deleted.

## Attempts to improve white wine model accuracy

White wine plsRglm model gives about 0.585 MAE. Let’s examine feature correlation of white wine dataset.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/white_feature_cor.png">
  <br>
  <i>Figure 7: Feature correlations, distributions and Pearson coefficients</i>
</p>

Numbers above diagonal defines Pearson correlation coefficients between corresponding features. From above plot matrix it can be clearly seen that some features are highly correlated with each other. It can bring to feature redundancy and consequently unstable model. Let's apply kernel PCA transformation to get linearly independent subset of all predictors.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/white_feature_cor_pca.png">
  <br>
  <i>Figure 8: Feature correlations, distributions and Pearson coefficients after PCA</i>
</p>

We got 9-dimensional subspace of predictors which are fully unrelated to each other. Training model for PCA transformed dataset slightly improved accuracy. Now there are 2 models trained for PCA transformed and raw datasets. Stacking them to another plsRglm model had relatively better accuracy having 0.535 MAE. Let’s test different 3 plsRglm models preprocessed by PCA, ICA and Box-Cox and 1 model on raw data.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/models_cor.png">
  <br>
  <i>Figure 9: Predictions correlation by 4 models trained on raw, PCA, ICA, BoxCox preprocessed datasets.</i>
</p>

From Figure 9 it can be concluded that stacking those models is not a good choice, since predictions they return are highly linearly correlated to each other (carry similar information). Stacking such models would not improve accuracy in average. Number of components in Independent component analysis was chosen empirically as 11 which always had best results in terms of MAE. One useful thing is that Principal Component analysis returns 9-dimensional subspace of 11 features having similar accuracy. So usage of PCA for preprocessing decreases number of features by 2 decreasing model complexity a little.

## Further attempt for model stacking

To decrease models’ correlation most correlated and uncorrelated features were chosen for 2 plsRglm models without data transformations.

<p align="center">
  <img width="75%" height="75%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/models_cor_unc.png">
  <br>
  <i>Figure 10: Correlation of 2 model predictions made on highly correlated and lowly correlated features.</i>
</p>

Now, models are not correlated very much, hence they could be stacked. However, each model individually had major decrease in accuracy for about 0.05-0.07 MAE. In general, stacking them would lead to performance of 1 plsRglm model trained on all features. Different subsets of features were tested for models to be stacked based on importance of features and correlation of them to each other. Also, they were preprocessed using 3 transformation methods used above. All tests’ results shown decrease in accuracy about 0.05-0.1. Ensembling transformed models leads to complicated model and frequently decrease in accuracy, therefore it will not be used. The only advantage of these tests was that training a model on PCA transformed dataset decreases dimensionality of predictors.

## Implementation of custom caret model

<p align="center">
  <img width="75%" height="75%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/pipeline.png">
  <br>
  <i>Figure 11: Model training pipeline</i>
</p>

Raw dataset is classified to red and white datasets. Then, each of the datasets are cleaned (outlier deletion) considering also output variable “quality”. Finally, for each cleaned dataset individual PlsRglm model is trained using independent parameters.

<p align="center">
  <img width="75%" height="75%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/pls_selection.png">
  <br>
  <i>Table 5: pls glm model selection for red and wine datasets</i>
</p>

PLS GLM models were chosen using grid tuning for 4 methods (pls-glm-Gamma, pls-glm-gaussian, pls-glm-inverse.gaussian, pls-glm-poisson). Best results were given by usage of pls-glm-poisson and pls-glm-inverse.gaussian for red and white datasets respectively. Other parameters (number of components and level of significance) were also slightly improved by tuning using 5 fold Cross-Validation. Best parameters are fixed in custom model, to change them tuneGrid parameter should be used in caret::train function. Transforming datasets using Principal-Component Analysis decreased training time as was expected (13 seconds) having MAE about 0.53. Training without PCA was little longer (18 seconds) having MAE about 0.52. This model is better than first proposed model in terms of accuracy (0.52 vs 0.57) and in terms of training time also (18 vs 39 seconds). Training time may differ on various machines.

<p align="center">
  <img width="75%" height="75%" src="https://github.com/BatyaGG/Wine-Quality-Score/blob/master/figures/histogram_error.png">
  <br>
  <i>Figure 10: Histogram of prediction error (difference between prediction and actual)</i>
</p>

Predictions never differs from actual value by more than 3. Mostly (more than 50%), quality is accurately predicted. About 40% observations are predicted with error of 1 quality score and about 10% data is predicted with 2 quality score difference.
