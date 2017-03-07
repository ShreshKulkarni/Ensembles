set.seed(345)
indFCShort <- sample(2,nrow(forestCovShort),replace = TRUE,prob = c(0.7,0.3))
trainFC <- forestCovShort[ind==1,]
testFC <- forestCovShort[ind==2,]

library(randomForest)
model.rf.base <- randomForest(Cover_Type ~. , data=trainForestCov,ntree=50,keep.inbag=TRUE,mtry=7,replace=TRUE)
model.rf.base
pred.rf.base <- predict(model.rf.base,testForestCov)
confusionMatrix(table(pred.rf.base,testForestCov$Cover_Type))

# Confusion Matrix and Statistics
# 
# 
# pred.rf.base     1     2     3     4     5     6     7
# 1 10081  1705     0     0    22     3   349
# 2  2540 15026   180     0   437   250    17
# 3     5   171  1914    75    24   499     0
# 4     0     1     6    92     0     1     0
# 5     0     5     0     0   105     0     0
# 6     1    25    22     0     2   287     0
# 7    60     6     0     0     0     0   895
# 
# Overall Statistics
# 
# Accuracy : 0.816         
# 95% CI : (0.8118, 0.82)
# No Information Rate : 0.4867        
# P-Value [Acc > NIR] : < 2.2e-16     
# 
# Kappa : 0.6976        
# Mcnemar's Test P-Value : NA            
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
# Sensitivity            0.7946   0.8871  0.90198 0.550898 0.177966 0.275962  0.70975
# Specificity            0.9060   0.8084  0.97632 0.999769 0.999854 0.998519  0.99803
# Pos Pred Value         0.8290   0.8144  0.71205 0.920000 0.954545 0.851632  0.93132
# Neg Pred Value         0.8849   0.8830  0.99352 0.997839 0.986021 0.978154  0.98919
# Prevalence             0.3645   0.4867  0.06097 0.004798 0.016951 0.029880  0.03623
# Detection Rate         0.2896   0.4317  0.05499 0.002643 0.003017 0.008246  0.02571
# Detection Prevalence   0.3494   0.5301  0.07723 0.002873 0.003160 0.009682  0.02761
# Balanced Accuracy      0.8503   0.8477  0.93915 0.775334 0.588910 0.637240  0.85389

trainCtrl <- trainControl(method="repeatedcv",repeats = 3,number=10)
set.seed(7)
metric <- "Accuracy"
mtry <- floor(sqrt(ncol(trainForestCov[,-55])))
tuneGrid <- expand.grid(.mtry = mtry)
model.rf.default <- train(Cover_Type ~., data=trainForestCov,trControl=trainCtrl,metric=metric,method="rf",tuneGrid = tuneGrid)
pred.rf.default <- predict(model.rf.default,testForestCov)
confusionMatrix(table(pred.rf.default,testForestCov$Cover_Type))
plot(model.rf.default)
# Confusion Matrix and Statistics
# 
# 
# pred.rf.default    X1    X2    X3    X4    X5    X6    X7
# X1 10224  1792     0     0    18     1   339
# X2  2402 14990   206     0   424   256     8
# X3     3   152  2013    70    17   505     0
# X4     0     2     7    80     0     8     0
# X5     1     7     0     0    91     0     0
# X6     0    11    22     0     0   323     0
# X7    58     6     0     0     0     0   863
# 
# Overall Statistics
# 
# Accuracy : 0.819          
# 95% CI : (0.815, 0.8231)
# No Information Rate : 0.486          
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.7034         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.8058    0.8838   0.89546  0.533333  0.165455  0.295517   0.71322
# Specificity             0.9032    0.8163   0.97712  0.999511  0.999767  0.999024   0.99810
# Pos Pred Value          0.8262    0.8198   0.72935  0.824742  0.919192  0.907303   0.93096
# Neg Pred Value          0.8906    0.8814   0.99269  0.997989  0.986810  0.977709   0.98979
# Prevalence              0.3636    0.4860   0.06441  0.004298  0.015760  0.031319   0.03467
# Detection Rate          0.2930    0.4295   0.05768  0.002292  0.002608  0.009255   0.02473
# Detection Prevalence    0.3546    0.5240   0.07909  0.002779  0.002837  0.010201   0.02656
# Balanced Accuracy       0.8545    0.8501   0.93629  0.766422  0.582611  0.647270   0.85566

trainCtrlRF <- trainControl(method = "repeatedcv",number = 10,repeats = 3,classProbs = TRUE,returnData = TRUE,
                            returnResamp = "all",verboseIter = TRUE)
model.rf.1 <- train(Cover_Type~.,data=trainFC,method="parRF",trControl=trainCtrlRF,tuneLength=5)
model.rf.1

# Parallel Random Forest 
# 
# 81303 samples
# 54 predictor
# 7 classes: 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 73174, 73173, 73173, 73173, 73174, 73172, ... 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2    0.6503856  0.3844221
# 15    0.9088799  0.8521189
# 28    0.9180842  0.8673466
# 41    0.9184982  0.8681007
# 54    0.9160588  0.8641937
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 41.

pred.test.rf.1 <- predict.train(model.rf.1,testFC)
confusionMatrix(table(pred.test.rf.1,testFC$Cover_Type))

# Confusion Matrix and Statistics
# 
# 
# pred.test.rf.1    X1    X2    X3    X4    X5    X6    X7
# X1 11521   686     0     0     6     2   122
# X2  1136 16166    69     0   183    53    10
# X3     1    62  2030    35    10   136     0
# X4     0     1    13   118     0     4     0
# X5     4    37     2     0   371     2     0
# X6     4    36    61     5     2   878     0
# X7    58     8     0     0     0     0  1067
# 
# Overall Statistics
# 
# Accuracy : 0.9213          
# 95% CI : (0.9184, 0.9241)
# No Information Rate : 0.487           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8728          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.9055    0.9512   0.93333  0.746835   0.64860   0.81674   0.88991
# Specificity             0.9632    0.9190   0.99254  0.999482   0.99869   0.99681   0.99804
# Pos Pred Value          0.9339    0.9176   0.89270  0.867647   0.89183   0.89047   0.94175
# Neg Pred Value          0.9467    0.9520   0.99556  0.998849   0.99417   0.99419   0.99609
# Prevalence              0.3646    0.4870   0.06232  0.004527   0.01639   0.03080   0.03436
# Detection Rate          0.3301    0.4632   0.05817  0.003381   0.01063   0.02516   0.03057
# Detection Prevalence    0.3535    0.5048   0.06516  0.003897   0.01192   0.02825   0.03247
# Balanced Accuracy       0.9343    0.9351   0.96294  0.873159   0.82365   0.90678   0.94397

varImp(model.rf.1)
# parRF variable importance
# 
# only 20 most important variables shown (out of 54)
# 
# Overall
# Elevation                          100.000
# Horizontal_Distance_To_Roadways     39.535
# Horizontal_Distance_To_Fire_Points  38.686
# Horizontal_Distance_To_Hydrology    20.248
# Vertical_Distance_To_Hydrology      16.862
# Hillshade_Noon                      14.531
# Aspect                              12.699
# Hillshade_9am                       11.695
# Hillshade_3pm                       10.828
# Slope                                8.279
# Wilderness_Area4                     8.261
# Wilderness_Area3                     3.934
# Soil_Type4                           3.804
# Soil_Type32                          3.500
# Wilderness_Area1                     3.468
# Soil_Type2                           3.262
# Soil_Type22                          3.022
# Soil_Type23                          2.803
# Soil_Type29                          2.031
# Soil_Type31                          1.738

  
