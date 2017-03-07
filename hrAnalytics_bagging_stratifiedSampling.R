library(doParallel)
library(e1071)
require(caret)

s <- summary(hrDS$left)
ind = sample(2,nrow(hrDS1),replace=TRUE,prob = c(0.8,0.2))
trainHR <- hrDS[ind==1,]
testHR <- hrDS[ind==2,]



model.base <- naiveBayes(left ~.,data=trainHR)
predict(model.base,testHR$left)

nIterations <- 50
div <- 5
predictions.bag <- foreach(1:nIterations,.combine = cbind) %do% {
  train_ind <- sample(nrow(trainHR),size = floor(nrow(trainHR)/div),replace = TRUE)
  train_pos <- 1:nrow(trainHR) %in% train_ind
  train_sample <- trainHR[train_pos,]
  model.train <- naiveBayes(left ~ .,data=train_sample)
  predict(model.train,testHR,type="class")
}

predictions <- round(rowMeans(predictions.bag))
class(predictions)
predictions <- factor(predictions,levels = c(1,2))
#predictions <- round(rowMeans(predictions.bag))
mean(predictions != testHR$left)
confusionMatrix(table(predictions,testHR$left))


predictions <- rowMeans(predictions.bag)
sqrt((sum((as.numeric(testHR$left)-predictions)^2))/nrow(testHR))


##stratified sampling
require(sampling)
## we will first take a stratified sample of 80% of data as training data and rest as test data
temp<-NULL
temp1<-NULL
temp <- hrDS1

##Manipulation on class left is done because ordering was producing greater no of 
#rows for Yes class as Yes comes after No alphabetically
temp$left <- as.numeric(temp$left)
temp$left <- factor(temp$left,levels = c(1,2))  

temp <- temp[order(temp[7]),]
size <- 0.7
size <- ceiling(table(temp[7])*size)
strat <- strata(temp,stratanames = names(temp[7]),size = size,method="srswr")
trainHR <- getdata(temp,strat)
testHR<- temp[-trainHR$ID_unit,]

##base model with stratified sampling
trainHR[,c(11,12,13)] <- NULL
testHR[,c(11,12,13)] <- NULL
model.nb.base <- naiveBayes(left ~.,data=trainHR)
pred.nb.base <- predict(model.nb.base,testHR$left,type="class")
t <- table(pred.nb.base,testHR$left)
confusionMatrix(t)


nIterations <- 50
div <- 5
predictions.bag <- foreach(1:nIterations,.combine = cbind) %do% {
  temp1 <- trainHR[order(trainHR[7]),]
  strat <- strata(temp1,stratanames = names(temp1[7]),size=ceiling(table(temp1[7])/div),method="srswr")
  train_sample <- getdata(temp1,strat)
  train_sample[,c(11,12,13)] <- NULL
  model.train <- naiveBayes(left ~ .,data=train_sample)
  predict(model.train,testHR,type="class")
}

predictions <- round(rowMeans(predictions.bag))
class(predictions)
predictions <- factor(predictions,levels = c(1,2))
#predictions <- round(rowMeans(predictions.bag))
mean(predictions != testHR$left)
confusionMatrix(table(predictions,testHR$left))


predictions <- rowMeans(predictions.bag)
sqrt((sum((as.numeric(testHR$left)-predictions)^2))/nrow(testHR))
