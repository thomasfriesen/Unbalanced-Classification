library(randomForest)
library(caret)
library(pRoc)
library(DMwR)
library(doParallel)
library(GGally)
library(ggfortify)
wine_red=read.csv("winequality-red.csv",header=T,sep=";")
wine_red$quality=factor(wine_red$quality)



ggplot(wine_red)+geom_boxplot(aes(x=quality,y=sulphates))
ggplot(wine_red)+geom_boxplot(aes(x=quality,y=sulphates))

ggpairs(wine_red,columns=1:4,aes(col=quality,alpha=0.8))


pca_wine=prcomp(wine_red[,1:11],center=T,scale=T)
autoplot(pca_wine,col="quality",data=wine_red)








##########################

table(wine_red$quality)
wine_red$binary=ifelse(wine_red$quality<=6,"G","B")
table(wine_red$binary)
wine_red$binary=factor(wine_red$binary)


set.seed(2508)
trainingRow=createDataPartition(wine_red$binary,p=0.7,list=F)
trainingSet=wine_red[trainingRow,]
testSet=wine_red[-trainingRow,]


#####################

tr_controll=trainControl(classProbs = T
                         ,method="repeatedcv"
                         ,savePredictions="all"
                         ,summaryFunction=defaultSummary
                         ,repeats=5
                         ,allowParallel=T)



library(doParallel)
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
model0=train(binary~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+alcohol+sulphates
             ,data=trainingSet,method="rf",metric="Kappa"
             ,maximize=F
             ,tuneLength=10
             ,ntree=1000
             ,trControl=tr_controll)

stopCluster(cl)
model0


Confusion_threshold=confusionMatrix(predict(model0,testSet),reference=testSet$binary)
Confusion_threshold

roc0=roc(testSet$binary,
    predict(model0, testSet, type = "prob")[,1],
    levels = rev(levels(as.factor(testSet$binary))))


ggroc(roc0)+geom_line(data=data.frame(x=seq(1,0,by=-0.01),y=seq(0,1,by=0.01)),aes(x=x,y=y))



##############

tr_controll=trainControl(classProbs = T
                         ,method="repeatedcv"
                         ,sampling="down"
                         ,savePredictions="all"
                         ,summaryFunction=defaultSummary
                         ,repeats=5
                         ,allowParallel=T)



library(doParallel)
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
model_down=train(binary~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+alcohol+sulphates
             ,data=trainingSet,method="rf",metric="Kappa"
             ,maximize=F
             ,tuneLength=10
             ,ntree=1000
             ,trControl=tr_controll)

stopCluster(cl)
model_down



Confusion_threshold_down=confusionMatrix(predict(model_down,testSet),reference=testSet$binary)
Confusion_threshold_down

roc_down=roc(testSet$binary,
         predict(model_down, testSet, type = "prob")[,1],
         levels = rev(levels(as.factor(testSet$binary))))


ggroc(roc_down)+geom_line(data=data.frame(x=seq(1,0,by=-0.01),y=seq(0,1,by=0.01)),aes(x=x,y=y))


#################

tr_controll=trainControl(classProbs = T
                         ,method="repeatedcv"
                         ,sampling="up"
                         ,savePredictions="all"
                         ,summaryFunction=defaultSummary
                         ,repeats=5
                         ,allowParallel=T)



library(doParallel)
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
model_up=train(binary~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+alcohol+sulphates
                 ,data=trainingSet,method="rf",metric="Kappa"
                 ,maximize=F
                 ,tuneLength=10
                 ,ntree=1000
                 ,trControl=tr_controll)

stopCluster(cl)
model_up



Confusion_threshold_up=confusionMatrix(predict(model_up,testSet),reference=testSet$binary)
Confusion_threshold_up

roc_up=roc(testSet$binary,
             predict(model_up, testSet, type = "prob")[,1],
             levels = rev(levels(as.factor(testSet$binary))))


ggroc(roc_up)+geom_line(data=data.frame(x=seq(1,0,by=-0.01),y=seq(0,1,by=0.01)),aes(x=x,y=y))




##############


tr_controll=trainControl(classProbs = T
                         ,method="repeatedcv"
                         ,sampling="smote"
                         ,savePredictions="all"
                         ,summaryFunction=defaultSummary
                         ,repeats=5
                         ,allowParallel=T)



cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
model_smote=train(binary~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+alcohol+sulphates
             ,data=trainingSet,method="rf",metric="Kappa"
             ,maximize=F
             ,tuneLength=10
             ,ntree=1000
             ,trControl=tr_controll)

stopCluster(cl)
model_smote


Confusion_threshold=confusionMatrix(predict(model_smote,testSet),reference=testSet$binary)
Confusion_threshold

roc_smote=roc(testSet$binary,
         predict(model_smote, testSet, type = "prob")[,1],
         levels = rev(levels(as.factor(testSet$binary))))


ggroc(roc_smote)+geom_line(data=data.frame(x=seq(1,0,by=-0.01),y=seq(0,1,by=0.01)),aes(x=x,y=y))

#####



##############

ggroc(list(Up=roc_up,Down=roc_down,Roc=roc0,SMOTE=roc_smote))+geom_line(data=data.frame(x=seq(1,0,by=-0.01),y=seq(0,1,by=0.01)),aes(x=x,y=y),inherit.aes=F)+
  ggtitle("ROC-Curve with different Sampling Methods")


#################
model_list=resamples(list(Up=model_up,Smote=model_smote,Down=model_down,No=model0))
model_diff=diff(model_list)
summary(model_diff)

dotplot(model_diff)




#############
var_down=varImp(model_down)
ggplot(var_down)+ggtitle("Variable Importance for Down-Sampling")

var_up=varImp(model_up)
ggplot(var_up)+ggtitle("Variable Importance for Up-Sampling")


var_smote=varImp(model_smote)
ggplot(var_smote)+ggtitle("Variable Importance for Smote-Sampling")

var_0=varImp(model0)
ggplot(var_0)+ggtitle("Variable Importance for no Sampling")







