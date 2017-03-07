library(caret)
library(doParallel)

registerDoParallel(cores = 4)
getDoParWorkers()

set.seed(345)
indFCShort <- sample(2,nrow(forestCovShort),replace = TRUE,prob = c(0.7,0.3))
trainFC <- forestCovShort[ind==1,]
testFC <- forestCovShort[ind==2,]

trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 3,classProbs = TRUE,returnResamp = "all"
                             ,verboseIter = TRUE)
tunegrid <- expand.grid(.mfinal = 50,.maxdepth = c(5:10))
print(Sys.time())
model.adabag.1 <- train(Cover_Type ~.,data=trainFC,method="AdaBag",tuneGrid=tunegrid,trControl=trainCtrlBag)
print(Sys.time())
pred.test.adabag <- predict.train(model.adabag.1,testFC)
confusionMatrix(table(pred.test.adabag,testFC$Cover_Type))

# Bagged AdaBoost 
# 
# 81303 samples
# 54 predictor
# 7 classes: 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 73173, 73171, 73173, 73173, 73172, 73174, ... 
# Resampling results across tuning parameters:
#   
#   maxdepth  Accuracy   Kappa    
# 5        0.7091825  0.5169502
# 6        0.7264430  0.5488901
# 7        0.7401654  0.5687697
# 8        0.7580244  0.5997831
# 9        0.7746044  0.6283187
# 10        0.7912705  0.6558812
# 
# Tuning parameter 'mfinal' was held constant at a value of 50
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were mfinal = 50 and maxdepth = 10. 
# > pred.test.adabag <- predict.train(model.adabag.1,testForestCov)
# > confusionMatrix(table(pred.test.adabag,testForestCov$Cover_Type))
# Confusion Matrix and Statistics
# 
# 
# pred.test.adabag    X1    X2    X3    X4    X5    X6    X7
# X1  9612  2061     0     0     5     1   364
# X2  3017 14726   235     1   437   275     6
# X3     1   101  1816    72    10   302     0
# X4     0     1    15    82     0     4     0
# X5     3    19     1     0   120     1     0
# X6     0    71   108     3     0   492     0
# X7    91    17     0     0     0     0   829
# 
# Overall Statistics
# 
# Accuracy : 0.7931          
# 95% CI : (0.7888, 0.7973)
# No Information Rate : 0.487           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.6594          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.7554    0.8664   0.83494  0.518987  0.209790   0.45767   0.69141
# Specificity             0.8904    0.7782   0.98515  0.999424  0.999301   0.99462   0.99680
# Pos Pred Value          0.7981    0.7876   0.78888  0.803922  0.833333   0.72997   0.88474
# Neg Pred Value          0.8638    0.8599   0.98899  0.997816  0.986995   0.98297   0.98911
# Prevalence              0.3646    0.4870   0.06232  0.004527  0.016390   0.03080   0.03436
# Detection Rate          0.2754    0.4220   0.05204  0.002350  0.003438   0.01410   0.02375
# Detection Prevalence    0.3451    0.5357   0.06596  0.002923  0.004126   0.01931   0.02685
# Balanced Accuracy       0.8229    0.8223   0.91005  0.759206  0.604546   0.72615   0.84410

tunegrid <- expand.grid(.mfinal = c(50,80,100,200,300),.maxdepth = c(5:20))
print(Sys.time())
model.adabag.2 <- train(Cover_Type ~.,data=trainFC,method="AdaBag",tuneGrid=tunegrid,trControl=trainCtrlBag)
print(Sys.time())
pred.adabag.2 <- predict.train(model.adabag.2,testForestCov)
confusionMatrix(table(pred.adabag.2,testFC$Cover_Type))