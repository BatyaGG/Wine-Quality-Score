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
