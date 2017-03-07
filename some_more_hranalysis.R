rootModName <- "mod_nb_class_"
extModName <- ".rda"
laplaceCorr <- 1
useKernEstim <- TRUE
#testHRmod <- testHR
modSel <- trainControl(method="cv",number=10)
modTun <- data.frame(.fL=laplaceCorr,.usekernel=useKernEstim,.adjust=2)
nTrain <- dim(trainHR)[1]
pObs <- rep(1/nTrain,nTrain)
Beta <- c()
nLearners <- 3
trainHR$left <- factor(as.numeric(trainHR$left),levels = c(1,2))
testHR$left <-factor(as.numeric(testHR$left),levels=c(1,2))

for(j in 1:nLearners) {
  nIterations <- 20
  div <- 5
#   predictions.bag <- foreach(1:nIterations,.combine = cbind) %do% {
#     train_ind <- sample(nrow(trainHR),size = floor(nrow(trainHR)/div),replace = TRUE)
#     train_pos <- 1:nrow(trainHR) %in% train_ind
#     train_sample <- trainHR[train_pos,]
#     model.train <- naiveBayes(left ~ .,data=train_sample)
#     predict(model.train,trainHR,type="class")
# #     train_ind <- sample(ncol(trainHR),size = floor(ncol(trainHR)/div),replace = TRUE)
# #     train_pos <- 1:ncol(trainHR) %in% train_ind
# #     if(!train_pos[7]) {
# #       train_pos[7] <- TRUE
# #     }
# #     train_sample <- trainHR[,train_pos]
# #     model.train <- naiveBayes(left ~ .,data=train_sample)
# #     predict(model.train,trainHR,type="class")
#     
#   }
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
  pred.nb.iter <- factor(predictions,levels = c(1,2))
  print(table(pred.nb.iter))
  # Calculating the error rate
  Ejt <- abs(as.numeric(pred.nb.iter) -
               as.numeric(trainHR$left))
  rm(pred.nb.iter)
  Ej <- sum(pObs*Ejt)
  
  # If error rate > 1/2, stop training with previously learned models
  if (Ej > 0.5) {
    nLearners <- j - 1
    print(paste("Adaptive learning finished. Error = ", Ej, "; number learners = ",
                nLearners, sep = ""))
    break
  }
  # Otherwise, learn weighting of the observations and save model
  else {
    # Calculating the learning rate
    Beta[j] <- Ej/(1.0 - Ej)
    
    # Reducing the significance only for correct predictions
    pObs <- (1.0 - Ejt)*Beta[j]*pObs + Ejt*pObs
    
    # Normalizing the probabilities of the observations
    pObs <- pObs/sum(pObs)
    
    # Saving the trained model in an R file
    thisModFile <- paste(rootModName, j, extModName, sep = "")
    save(model.train, file = thisModFile)
    rm(model.train)
  }
}

# Testing the model with AdaBoost.M1
nTest <- dim(testHR)[1]
pred.nb.wts <- rep(0, nTest)

for (j in 1:nLearners){
  # Re-load learned model
  thisModFile <- paste(rootModName, j, extModName, sep = "")
  load(thisModFile)
  
  # Predict learned model
  predNBClass <- predict(model.train, testHR)
  rm(model.train)
  print(predNBClass)
  
  # Combining the models
  if (nLearners >= 2) {
    # Predict with weighting of the learned model
    pred.nb.wts <- pred.nb.wts +
      log(1.0/Beta[j])*as.numeric(predNBClass)
  }
  else {
    # Non-boosted predictions
    pred.nb.wts <- as.numeric(predNBClass)
  }
  rm(predNBClass)
}

# Standardizing the predictions if various models
if (nLearners >= 2) {
  pred.nb.wts <- pred.nb.wts/sum(log(1.0/Beta))
}

# Re-implementing in categories
totPredNBClass <- as.factor(as.integer(pred.nb.wts ))

# Show the probability matrix
print(nLearners)
#t<-table(factor(as.numeric(totPredNBClass),levels=c(1,2)),testHR$left )
t<-table(totPredNBClass,testHR$left)
confusionMatrix(t)


nIterations <- 5
div <- 3
predictions.bag <- foreach(1:nIterations,.combine = cbind) %do% {
  train_ind <- sample(ncol(trainHR),size = floor(ncol(trainHR)/div),replace = TRUE)
  train_pos <- 1:ncol(trainHR) %in% train_ind
  if(!train_pos[7]) {
    train_pos[7] <- TRUE
  }
  train_sample <- trainHR[,train_pos]
  model.train <- naiveBayes(left ~ .,data=train_sample)
  predict(model.train,testHR,type="class")
}

predictions <- round(rowMeans(predictions.bag))
class(predictions)
pred.nb.iter <- factor(predictions,levels = c(1,2))
confusionMatrix(table(pred.nb.iter,testHR$left))
