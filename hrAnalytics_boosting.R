library(e1071)
library(caret)
library(MASS)
library(lattice)

hrDS <- read.csv("~/MSPA/Kaggle/Datasets/HR_comma_sep.csv")
summary(hrDS)
hrDS$left <- factor(hrDS$left,levels = c(1,0),labels = c("Yes","No"))
hrDS$promotion_last_5years <-factor(hrDS$promotion_last_5years,levels = c(1,0),labels = c("Yes","No"))
hrDS$Work_accident <- factor(hrDS$Work_accident,levels = c(1,0),labels = c("Yes","No"))

ind <- sample(2,nrow(hrDS),replace = TRUE,prob = c(0.8,0.2))
trainHR <- hrDS[ind==1,]
testHR <- hrDS[ind==2,]

rootModName <- "mod_nb_class_"
extModName <- ".rda"
laplaceCorr <- 1
useKernEstim <- TRUE
#testHRmod <- testHR
modSel <- trainControl(method="cv",number=10)
modTun <- data.frame(.fL=laplaceCorr,.usekernel=useKernEstim,.adjust=2)
model.nb.base <- train(left ~ .,data=trainHR,method="nb",trControl=modSel)
pred.nb.base <- predict(model.nb.base,testHR)
model.nb.base1 <- naiveBayes(left ~., data=trainHR)

nTrain <- dim(trainHR)[1]
pObs <- rep(1/nTrain,nTrain)
Beta <- c()
nLearners <- 5
trainHR$left <- factor(as.numeric(trainHR$left),levels = c(1,2))
testHR$left <-factor(as.numeric(testHR$left),levels=c(1,2))

for(j in 1:nLearners) {
  wtdInd <- sample(seq(1:nTrain),nTrain,replace = TRUE,prob = pObs)
  wtdTrainSample <- trainHR[wtdInd,]
  
  model.nb.iter <- naiveBayes(left ~.,data=wtdTrainSample,laplace = TRUE)
  #model.nb.iter <- train(left ~.,data=wtdTrainSample,method="nb",trControl=modSel,tuneGrid=modTun)
  pred.nb.iter <- predict(model.nb.iter, trainHR)
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
    save(model.nb.iter, file = thisModFile)
    rm(model.nb.iter)
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
  predNBClass <- predict(model.nb.iter, testHR)
  rm(model.nb.iter)
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
