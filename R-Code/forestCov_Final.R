### Classification of Forest Cover type based on the given parameters
library(caTools)
library(rpart)
library(caret)

forestCovDS <- read.csv("~/MSPA/Kaggle/Datasets/covtype.csv")

##Let's do some initial data analysis
summary(forestCovDS)

#We see above that dataset is pretty clean and Soil_Type and Wilderness_Area are
#categorical variables already converted to dummy variables, so we are good to 
#go in that respect.All we need to is convert the class variable Cover_type to 
#factor and preprocess the numeric variables. But before that lets look at some
#of the graphs to understand the relationship between various variables.

corMatrix <- cor(forestCovDS[,c(1:10,55)])
corrplot::corrplot(corMatrix,method = "color",type = "lower")

# forestCovDS$Cover_Type = factor(forestCovDS$Cover_Type,
#                                    levels=c(1,2,3,4,5,6,7),labels=c("X1","X2","X3","X4","X5","X6","X7"))

forestCovDS[,1:10] <- scale(forestCovDS[,1:10] )

#Let's take a very small stratified sample to see the performance of rpart on  it
set.seed(9087)
ind = sample.split(forestCovDS$Cover_Type,SplitRatio = 0.2)
forestCovShort <- forestCovDS[ind,]
forestCovShort$Cover_Type = factor(forestCovShort$Cover_Type,levels=c(1,2,3,4,5,6,7),labels=c("X1","X2","X3","X4","X5","X6","X7"))
table(forestCovShort$Cover_Type)

#Let's divide the short sample into test and train sets
set.seed(1234)
ind <- sample(2,nrow(forestCovShort),replace=TRUE,prob = c(0.7,0.3))
trainForestCov <- forestCovShort[ind==1,]
testForestCov <- forestCovShort[ind==2,]

#Basic Decision tree model

tree.base.short <- rpart(Cover_Type ~ .,data=trainForestCov)
pred.test.base.short <- predict(tree.base.short,testForestCov,type="class")
confusionMatrix(table(pred.test.base.short,testForestCov$Cover_Type))

# Confusion Matrix and Statistics
# 
# 
# pred.test.base.short     1     2     3     4     5     6     7
# 1  9366  4319     0     0     0     0  1181
# 2  3356 12737   604     0   575   308     5
# 3     1   225  1542   176     8   722     0
# 4     0     0     0     0     0     0     0
# 5     0     0     0     0     0     0     0
# 6     0     0     0     0     0     0     0
# 7     0     0     0     0     0     0     0
# 
# Overall Statistics
# 
# Accuracy : 0.6732          
# 95% CI : (0.6682, 0.6781)
# No Information Rate : 0.492           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4514          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
# Sensitivity            0.7361   0.7371  0.71855 0.000000   0.0000  0.00000  0.00000
# Specificity            0.7545   0.7283  0.96568 1.000000   1.0000  1.00000  1.00000
# Pos Pred Value         0.6300   0.7243  0.57666      NaN      NaN      NaN      NaN
# Neg Pred Value         0.8343   0.7409  0.98139 0.994989   0.9834  0.97068  0.96623
# Prevalence             0.3622   0.4920  0.06110 0.005011   0.0166  0.02932  0.03377
# Detection Rate         0.2666   0.3626  0.04390 0.000000   0.0000  0.00000  0.00000
# Detection Prevalence   0.4232   0.5006  0.07613 0.000000   0.0000  0.00000  0.00000
# Balanced Accuracy      0.7453   0.7327  0.84211 0.500000   0.5000  0.50000  0.50000

#Evaluate the result of resampling on the basic model.
trainCtrl <- trainControl(method="repeatedcv",repeats=3,classProbs = TRUE,returnResamp = "all")
model.train.forest.short <- train(Cover_Type~.,data=trainForestCov,method="rpart",trControl=trainCtrl,metric="Accuracy")

#This is done because the class spit out by the model is of the form X1, X2...
testForestCov$Cover_Type<- factor(testForestCov$Cover_Type,levels=levels(testForestCov$Cover_Type),labels = make.names(levels(testForestCov$Cover_Type)))

pred.test.model.train <- predict(model.train.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.train,testForestCov$Cover_Type))
# Confusion Matrix and Statistics
# 
# 
# pred.test.model.train    X1    X2    X3    X4    X5    X6    X7
# X1  9366  4319     0     0     0     0  1181
# X2  3356 12737   604     0   575   308     5
# X3     1   225  1542   176     8   722     0
# X4     0     0     0     0     0     0     0
# X5     0     0     0     0     0     0     0
# X6     0     0     0     0     0     0     0
# X7     0     0     0     0     0     0     0
# 
# Overall Statistics
# 
# Accuracy : 0.6732          
# 95% CI : (0.6682, 0.6781)
# No Information Rate : 0.492           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4514          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.7361    0.7371   0.71855  0.000000    0.0000   0.00000   0.00000
# Specificity             0.7545    0.7283   0.96568  1.000000    1.0000   1.00000   1.00000
# Pos Pred Value          0.6300    0.7243   0.57666       NaN       NaN       NaN       NaN
# Neg Pred Value          0.8343    0.7409   0.98139  0.994989    0.9834   0.97068   0.96623
# Prevalence              0.3622    0.4920   0.06110  0.005011    0.0166   0.02932   0.03377
# Detection Rate          0.2666    0.3626   0.04390  0.000000    0.0000   0.00000   0.00000
# Detection Prevalence    0.4232    0.5006   0.07613  0.000000    0.0000   0.00000   0.00000
# Balanced Accuracy       0.7453    0.7327   0.84211  0.500000    0.5000   0.50000   0.50000


#Let's try to run the bagging algorithm, we will use treebag from caret package
#we will also use doParallel library to use parallel computation
library(doParallel)
registerDoParallel(cores=4)
getDoParWorkers()

model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag")
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))
# Confusion Matrix and Statistics
# 
# 
# pred.test.model.bag    X1    X2    X3    X4    X5    X6    X7
# X1 11241   983     0     0    19     5   129
# X2  1387 16103    69     0   204    76    13
# X3     1    70  1920    32    12   155     0
# X4     0     0    28   136     0     7     0
# X5     6    65     7     0   346     3     0
# X6     4    52   122     8     2   784     0
# X7    84     8     0     0     0     0  1044
# 
# Overall Statistics
# 
# Accuracy : 0.8989         
# 95% CI : (0.8957, 0.902)
# No Information Rate : 0.492          
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.836          
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.8835    0.9318   0.89469  0.772727  0.593482   0.76117   0.88027
# Specificity             0.9493    0.9020   0.99181  0.998999  0.997655   0.99449   0.99729
# Pos Pred Value          0.9082    0.9020   0.87671  0.795322  0.810304   0.80658   0.91901
# Neg Pred Value          0.9349    0.9318   0.99314  0.998856  0.993170   0.99280   0.99582
# Prevalence              0.3622    0.4920   0.06110  0.005011  0.016598   0.02932   0.03377
# Detection Rate          0.3200    0.4584   0.05466  0.003872  0.009851   0.02232   0.02972
# Detection Prevalence    0.3524    0.5082   0.06235  0.004868  0.012157   0.02767   0.03234
# Balanced Accuracy       0.9164    0.9169   0.94325  0.885863  0.795569   0.87783   0.93878

#Let's try cross validation to verify the accuracy improvements we saw in previous step.
trainCtrlBag <- trainControl(method="cv",repeats = 3,classProbs = TRUE,returnResamp = "all")
model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag",trControl=trainCtrlBag)
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))
# # Confusion Matrix and Statistics
# # 
# # 
# # pred.test.model.bag    X1    X2    X3    X4    X5    X6    X7
# # X1 11417   799     0     0    15     1   118
# # X2  1230 16322    72     0   189    72    15
# # X3     0    51  1956    23    16   135     0
# # X4     0     0    18   146     0     6     0
# # X5     5    57     2     0   359     2     0
# # X6     1    49    98     7     4   814     0
# # X7    70     3     0     0     0     0  1053
# # 
# # Overall Statistics
# # 
# # Accuracy : 0.9129          
# # 95% CI : (0.9099, 0.9159)
# # No Information Rate : 0.492           
# # P-Value [Acc > NIR] : < 2.2e-16       
# # 
# # Kappa : 0.8587          
# # Mcnemar's Test P-Value : NA              
# # 
# # Statistics by Class:
# # 
# # Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# # Sensitivity             0.8974    0.9445   0.91146  0.829545   0.61578   0.79029   0.88786
# # Specificity             0.9584    0.9116   0.99318  0.999313   0.99809   0.99534   0.99785
# # Pos Pred Value          0.9245    0.9118   0.89684  0.858824   0.84471   0.83659   0.93517
# # Neg Pred Value          0.9427    0.9443   0.99423  0.999142   0.99354   0.99368   0.99609
# # Prevalence              0.3622    0.4920   0.06110  0.005011   0.01660   0.02932   0.03377
# # Detection Rate          0.3250    0.4647   0.05569  0.004157   0.01022   0.02317   0.02998
# # Detection Prevalence    0.3516    0.5096   0.06209  0.004840   0.01210   0.02770   0.03206
# Balanced Accuracy       0.9279    0.9280   0.95232  0.914429   0.80693   0.89281   0.94285


#Let's try repeatedcv for repeated resampling and see if it can bring improvement to the recall of 
#under-represented classes
trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 3,classProbs = TRUE,returnResamp = "all")
model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag",trControl=trainCtrlBag)
 pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))
table(testForestCov$Cover_Type)
# Confusion Matrix and Statistics
# 
# 
# pred.test.model.bag    X1    X2    X3    X4    X5    X6    X7
# X1 11416   803     2     0    14     1   118
# X2  1230 16310    55     1   189    73    19
# X3     0    58  1980    23    14   134     0
# X4     0     0    19   144     0     5     0
# X5     5    64     3     0   362     2     0
# X6     4    40    87     8     4   815     0
# X7    68     6     0     0     0     0  1049
# 
# Overall Statistics
# 
# Accuracy : 0.9132          
# 95% CI : (0.9102, 0.9161)
# No Information Rate : 0.492           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8591          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.8973    0.9438   0.92265  0.818182   0.62093   0.79126   0.88449
# Specificity             0.9581    0.9122   0.99306  0.999313   0.99786   0.99581   0.99782
# Pos Pred Value          0.9241    0.9123   0.89633  0.857143   0.83028   0.85073   0.93411
# Neg Pred Value          0.9426    0.9437   0.99496  0.999085   0.99363   0.99371   0.99597
# Prevalence              0.3622    0.4920   0.06110  0.005011   0.01660   0.02932   0.03377
# Detection Rate          0.3250    0.4643   0.05637  0.004100   0.01031   0.02320   0.02986
# Detection Prevalence    0.3517    0.5090   0.06289  0.004783   0.01241   0.02727   0.03197
# Balanced Accuracy       0.9277    0.9280   0.95785  0.908748   0.80939   0.89353   0.94115

#repeatedcv with repeats=5
trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 5,classProbs = TRUE,returnResamp = "all")
model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag",trControl=trainCtrlBag)
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))
# Confusion Matrix and Statistics
# 
# 
# pred.test.model.bag    X1    X2    X3    X4    X5    X6    X7
# X1 11412   787     1     0    18     1   109
# X2  1230 16342    67     0   182    75    17
# X3     0    42  1956    24    16   122     0
# X4     0     0    25   144     0     4     0
# X5     5    56     4     0   363     1     0
# X6     6    48    93     8     4   827     0
# X7    70     6     0     0     0     0  1060
# 
# Overall Statistics
# 
# Accuracy : 0.914          
# 95% CI : (0.911, 0.9169)
# No Information Rate : 0.492          
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.8604         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.8970    0.9457   0.91146  0.818182   0.62264   0.80291   0.89376
# Specificity             0.9591    0.9120   0.99381  0.999170   0.99809   0.99534   0.99776
# Pos Pred Value          0.9257    0.9123   0.90556  0.832370   0.84615   0.83874   0.93310
# Neg Pred Value          0.9425    0.9454   0.99424  0.999084   0.99366   0.99405   0.99629
# Prevalence              0.3622    0.4920   0.06110  0.005011   0.01660   0.02932   0.03377
# Detection Rate          0.3249    0.4653   0.05569  0.004100   0.01033   0.02354   0.03018
# Detection Prevalence    0.3510    0.5100   0.06149  0.004925   0.01221   0.02807   0.03234
# Balanced Accuracy       0.9280    0.9288   0.95264  0.908676   0.81037   0.89912   0.94576

#repeatedcv with repeats=10
trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 10,classProbs = TRUE,returnResamp = "all")
model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag",trControl=trainCtrlBag)
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))

# Confusion Matrix and Statistics
# 
# 
# pred.test.model.bag    X1    X2    X3    X4    X5    X6    X7
# X1 11385   802     0     0    16     1   124
# X2  1264 16314    66     1   187    64    14
# X3     1    54  1966    27    15   130     0
# X4     0     0    17   141     0     5     0
# X5     6    60     4     0   360     0     0
# X6     4    45    93     7     5   830     0
# X7    63     6     0     0     0     0  1048
# 
# Overall Statistics
# 
# Accuracy : 0.9123          
# 95% CI : (0.9093, 0.9152)
# No Information Rate : 0.492           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8576          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: X1 Class: X2 Class: X3 Class: X4 Class: X5 Class: X6 Class: X7
# Sensitivity             0.8948    0.9440   0.91612  0.801136   0.61750   0.80583   0.88364
# Specificity             0.9579    0.9106   0.99312  0.999371   0.99797   0.99548   0.99797
# Pos Pred Value          0.9235    0.9109   0.89649  0.865031   0.83721   0.84350   0.93823
# Neg Pred Value          0.9413    0.9438   0.99453  0.998999   0.99357   0.99414   0.99594
# Prevalence              0.3622    0.4920   0.06110  0.005011   0.01660   0.02932   0.03377
# Detection Rate          0.3241    0.4645   0.05597  0.004014   0.01025   0.02363   0.02984
# Detection Prevalence    0.3510    0.5099   0.06243  0.004641   0.01224   0.02801   0.03180
# Balanced Accuracy       0.9264    0.9273   0.95462  0.900253   0.80773   0.90065   0.94080

#repeatedcv with down sampling
trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 3,classProbs = TRUE,returnResamp = "all",sampling = "down")
model.bag.forest.short <- train(Cover_Type ~.,data=trainForestCov,method="treebag",trControl=trainCtrlBag)
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))
# Bagged CART 
# 
# 81303 samples
# 54 predictor
# 7 classes: 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 73172, 73173, 73173, 73176, 73173, 73172, ... 
# Addtional sampling using down-sampling
# 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.6195915  0.4537971

seeds <- sample.int(80000,10)
pred.bag <- foreach(j=1:5,.combine=rbind) %dopar% {
  library(caret)
trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 3,classProbs = TRUE,returnResamp = "all"
                             ,sampling = "up",verboseIter = TRUE)

set.seed(seeds[j])
traindataind <- sample(1:nrow(trainForestCov),size=floor(nrow(trainForestCov)*0.2))
traindata <- trainForestCov[traindataind,]
train(Cover_Type ~., data=traindata,method="treebag",trControl=trainCtrlBag)
}
pred.test.model.bag <- predict.train(model.bag.forest.short,testForestCov)
confusionMatrix(table(pred.test.model.bag,testForestCov$Cover_Type))





#Let's try to scale the final selected model to complete dataset. since, it is bagging we can parallelize the process 
#easily by splitting the training sets into 10 sets of 10% each and collecting the results with majority voting.
#First apply the preprocessing steps
forestCovDS[,1:10] <- scale(forestCovDS[,1:10])
forestCovDS$Cover_Type <- factor(forestCovDS$Cover_Type,
                                      levels=c(1,2,3,4,5,6,7),labels=c("X1","X2","X3","X4","X5","X6","X7"))

#Division of dataset into test and train
set.seed(98765)
indF <- sample(2,nrow(forestCovDS),replace = TRUE,prob = c(0.7,0.3))
finalTrainFC <-forestCovDS[indF==1,]
finalTestFC <- forestCovDS[indF==2,]

seeds <- sample.int(400000,10)
models <- c()
# rm(models,model.j,pred.bag)
# rm(testForestCov,trainForestCov,traindata,pred.final.test,traindataind)
pred.bag <- foreach(j=1:10) %do% {
  library(caret)
  trainCtrlBag <- trainControl(method="repeatedcv",number=10,repeats = 3,classProbs = TRUE,returnResamp = "all"
                               ,verboseIter = TRUE)
  
  set.seed(seeds[j])
  traindataind <- sample(1:nrow(finalTrainFC),size=floor(nrow(finalTrainFC)*0.1))
  traindata <- finalTrainFC[traindataind,]
  models[j] <- train(Cover_Type ~., data=traindata,method="treebag",trControl=trainCtrlBag)
}

rm(traindata)
#extract prediction on test set from combination of 10 models
pred.final.test <- extractPrediction(pred.bag,finalTestFC[,1:54],finalTestFC[,55],verbose = TRUE)

#Extract the test predictions only since the prediction set from previous step contains predictions for
#training set as well.
pred.test.com <- cbind(pred.final.test[pred.final.test$object=="Object1" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object2" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object3" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object4" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object5" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object6" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object7" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object8" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object9" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object10" & pred.final.test$dataType=="Test",]$pred,
      pred.final.test[pred.final.test$object=="Object1" & pred.final.test$dataType=="Test",]$obs)

colnames(pred.test.com)[11] <- "obs"

for (i in 1:nrow(pred.test.com)) {
  pred.test.com[i,12] <- as.numeric(names(which.max(table(pred.test.com[i,1:10]))))
}

pred.test.com <- as.data.frame(pred.test.com)
pred.test.com$finalPred <- pred.test.mv
confusionMatrix(table(pred.test.com$finalPred,pred.test.com$obs))

library(reshape2)
plotdata <- melt(pred.test.com[,11:12])


ggplot(data=plotdata,aes(value,fill=variable))+geom_bar(position = "dodge")
#pred.test.bag.2 <- predict.train(model.bag.forest.short,forestCovDS)
#confusionMatrix(table(pred.test.bag.2,forestCovDS$Cover_Type))

#fitting on all data
pred.final.test <- extractPrediction(pred.bag,forestCovDS[,1:54],forestCovDS[,55],verbose = TRUE)
pred.final.com <- cbind(pred.final.test[pred.final.test$object=="Object1" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object2" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object3" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object4" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object5" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object6" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object7" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object8" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object9" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object10" & pred.final.test$dataType=="Test",]$pred,
                       pred.final.test[pred.final.test$object=="Object1" & pred.final.test$dataType=="Test",]$obs)


colnames(pred.final.com)[11] <- "obs"
pred.final.mv <- c()
for (i in 1:nrow(pred.final.com)) {
  pred.final.mv[i] <- as.numeric(names(which.max(table(pred.final.com[i,1:10]))))
}

pred.final.com <- as.data.frame(pred.final.com)
pred.final.com$finalPred <- pred.final.mv
confusionMatrix(table(pred.final.com$finalPred,pred.final.com$obs))
# Confusion Matrix and Statistics
# 
# 
# 1      2      3      4      5      6      7
# 1 191275  15050      2      0    195     21   2894
# 2  19582 266375   1687      1   4175   1296    131
# 3     11    904  33066    550    158   2641      0
# 4      0      7    164   2130      0     77      0
# 5     61    330     10      0   4946     12      1
# 6     49    553    825     66     19  13320      0
# 7    862     82      0      0      0      0  17484
# 
# Overall Statistics
# 
# Accuracy : 0.9098         
# 95% CI : (0.909, 0.9105)
# No Information Rate : 0.4876         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.8537         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
# Sensitivity            0.9029   0.9403  0.92482 0.775391 0.521015  0.76697  0.85246
# Specificity            0.9508   0.9097  0.99218 0.999571 0.999276  0.99732  0.99832
# Pos Pred Value         0.9133   0.9084  0.88578 0.895711 0.922761  0.89806  0.94877
# Neg Pred Value         0.9447   0.9412  0.99506 0.998934 0.992101  0.99285  0.99462
# Prevalence             0.3646   0.4876  0.06154 0.004728 0.016339  0.02989  0.03530
# Detection Rate         0.3292   0.4585  0.05691 0.003666 0.008513  0.02293  0.03009
# Detection Prevalence   0.3605   0.5047  0.06425 0.004093 0.009225  0.02553  0.03172
# Balanced Accuracy      0.9269   0.9250  0.95850 0.887481 0.760146  0.88214  0.92539


varImp(model.bag.forest.short)
# treebag variable importance
# 
# only 20 most important variables shown (out of 54)
# 
# Overall
# Elevation                          100.000
# Horizontal_Distance_To_Roadways     60.711
# Horizontal_Distance_To_Fire_Points  59.743
# Horizontal_Distance_To_Hydrology    42.844
# Vertical_Distance_To_Hydrology      38.866
# Aspect                              34.738
# Hillshade_9am                       32.678
# Hillshade_Noon                      32.083
# Hillshade_3pm                       30.284
# Wilderness_Area4                    28.117
# Slope                               26.881
# Soil_Type22                         13.726
# Wilderness_Area1                    12.881
# Soil_Type10                         12.791
# Soil_Type23                          9.812
# Wilderness_Area3                     8.119
# Soil_Type32                          7.819
# Soil_Type12                          7.027
# Soil_Type31                          4.346
# Soil_Type39                          4.008
rs<-resamples(pred.bag)
summary(rs)
trellis.par.set(caretTheme())
bwplot(rs,layout=c(1,1))