# clearing workspace and evaluating script directory
rm(list=ls())
script_dir = dirname(sys.frame(1)$ofile)

# importing dependencies
library(caret)

# extending caret plsRglm model
custom_model <- list(library = "plsRglm",
                     loop = NULL,
                     type = "Regression",
                     
                     # Parameters of custom model, their types and labels
                     parameters = data.frame(parameter = c("nt.red", "alpha.pvals.expli.red",
                                                           "nt.white", "alpha.pvals.expli.white",
                                                           "modele.red", "modele.white", "pca.prep"),
                                             class = c(rep("numeric", 4), rep("character", 2), "logical"),
                                             label = c("#PLS Components (red)", "p-Value threshold (red)",
                                                       "#PLS Components (white)", "p-Value threshold (white)",
                                                       "PLS Model (red)", "PLS Model (white)", "PCA preprocessed")),
                     
                     # Parameter estimation function (fixed to specific values found beforehand)
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
                     
                     # model train function
                     fit = function(x, y, wts, param, lev, last, classProbs, ...) {
                       
                       # importing custom functions for wine type classifications 
                       # and cleaning data from outliers
                       source(paste(script_dir, "clear_data.R", sep = "/"))
                       source(paste(script_dir, "wine_type.R", sep = "/"))
                       
                       # classification of red/white wine datasets and cleaning them independently
                       is_red <- wine_type(x)
                       data <- data.frame(x, y)
                       data_red <- data[is_red,]
                       data_white <- data[!is_red,]
                       data_red <- clear_data(data_red)
                       data_white <- clear_data(data_white)
                       
                       # input/output data retrieval for red/white wine datasets
                       x_red <- data_red[,-(ncol(x) + 1)]
                       y_red <- data_red[,(ncol(x) + 1)]
                       x_white <- data_white[,-(ncol(x) + 1)]
                       y_white <- data_white[,(ncol(x) + 1)]
                       
                       if (param$pca.prep){
                         # PCA transformation
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
                       
                       # return object for data not preprocessed by PCA
                       mod <- list(pca_prep=param$pca.prep, red=mod_red, white=mod_white)
                       
                       if (param$pca.prep){
                         # return object for PCA preprocessed data
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

# reading red/white wine data and concatenating
data_red = read.csv(paste(script_dir, "winequality-red.csv", sep = "/"), header = TRUE, sep = ";")
data_white = read.csv(paste(script_dir, "winequality-white.csv", sep = "/"), header = TRUE, sep = ";")
data <- rbind(data_red, data_white)

# training/testing a model with 5 fold cross-validation
folds <- 5
train_control = trainControl(method = "cv", number = folds)
start.time <- Sys.time()
model <- train(y = data[, 12], x = data[, -12],
                   method = custom_model, trControl = train_control)
message(folds, " fold CV tested model MAE is: ", model[[4]]$MAE, 
        "\nExecution time is: ", Sys.time() - start.time, " seconds")
