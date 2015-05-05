library("klaR")
library("caret")
library("ggplot2")
library("e1071")
library(klaR)

trainingdata10 = read.csv("kddcup.data_10_percentUnique",header=F)
testdata = read.csv("kddcup.testdata.corrected",header=F)
trainingdata10h = read.csv("kddcup.data_10_percentUniqueHeader.csv",header=T)

testdata10 = unique(testdata)

#Naive bayes model
nbmodel = naiveBayes(factor(V42)~.,data=trainingdata10)
nbtR001= sum(as.character(predict(nbmodel,trainingdata10[-42],threshold=0.01)) == trainingdata10$V42)/length(trainingdata10$V42)
nbTesU001 = sum(as.character(predict(nbmodel,testdata10[-42],threshold=0.01)) == testdata10$V42)/length(testdata10$V42)


nbmodel = naiveBayes(factor(classLabel)~.,data=trainingdata10h)
nbTesU001 = sum(as.character(predict(nbmodel,testdata10[-42],threshold=0.5)) == testdata10$V42)/length(testdata10$V42)

nbtR0= sum(as.character(predict(nbmodel,trainingdata10[-42])) == trainingdata10$V42)/length(trainingdata10$V42)
nbTesU0 = sum(as.character(predict(nbmodel,testdata10[-42])) == testdata10$V42)/length(testdata10$V42)

nbtR01= sum(as.character(predict(nbmodel,trainingdata10[-42],threshold=0.1)) == trainingdata10$V42)/length(trainingdata10$V42)
nbTesU01 = sum(as.character(predict(nbmodel,testdata10[-42],threshold=0.1)) == testdata10$V42)/length(testdata10$V42)

nbtR05= sum(as.character(predict(nbmodel,trainingdata10[-42],threshold=0.5)) == trainingdata10$V42)/length(trainingdata10$V42)
nbTesU05 = sum(as.character(predict(nbmodel,testdata10[-42],threshold=0.5)) == testdata10$V42)/length(testdata10$V42)

nbtR1= sum(as.character(predict(nbmodel,trainingdata10[-42],threshold=1)) == trainingdata10$V42)/length(trainingdata10$V42)
nbTesU1 = sum(as.character(predict(nbmodel,testdata10[-42],threshold=1)) == testdata10$V42)/length(testdata10$V42)


#kNN model

library("FNN")

knnResult3 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=3)
knnResult5 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=5)
knnResult10 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=10)
knnResult30 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=30)
knnResult100 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=100)
knnResult300 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=300)

knnR3 = sum(as.character(knnResult3) == trainindata10[,42])/length(knnResult3)
knnR5 = sum(as.character(knnResult5) == testdata10[,42])/length(knnResult3)
knnR10 = sum(as.character(knnResult10) == testdata10[,42])/length(knnResult3)
knnR30 = sum(as.character(knnResult30) == testdata10[,42])/length(knnResult30)
knnR100 = sum(as.character(knnResult100) == testdata10[,42])/length(knnResult30)

knnResultTr3 = knn(trainingdata10[c(-2,-3,-4,-42)],testdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=3)
knnR3 = sum(as.character(knnResultTr3) == testdata10[,42])/length(knnResultTr3)


knnResultTr3 = knn(trainingdata10[c(-2,-3,-4,-42)],trainingdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=3)
knnR3 = sum(as.character(knnResultTr3) == trainingdata10[,42])/length(knnResultTr3)
knnResultTr5 = knn(trainingdata10[c(-2,-3,-4,-42)],trainingdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=5)
knnR5 = sum(as.character(knnResultTr5) == trainingdata10[,42])/length(knnResultTr3)
knnResultTr10 = knn(trainingdata10[c(-2,-3,-4,-42)],trainingdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=10)
knnR10 = sum(as.character(knnResultTr10) == trainingdata10[,42])/length(knnResultTr3)
knnResultTr30 = knn(trainingdata10[c(-2,-3,-4,-42)],trainingdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=30)
knnR30 = sum(as.character(knnResultTr30) == trainingdata10[,42])/length(knnResultTr3)
knnResultTr100 = knn(trainingdata10[c(-2,-3,-4,-42)],trainingdata10[c(-2,-3,-4,-42)],factor(trainingdata10$V42),k=100)

#plot correlation 
library(corrplot)
corcsv= read.csv("cor.csv", row.names=1)
ct= corcsv[c(-1)]
rk =corcsv[c(1)]
rownames(ct) <- rk
par(mar = c(0.2, 0.3, 0.4, 0.5))
corrplot(data.matrix(corcsv), mar=c(1,1,1,1),tl.cex = 0.7,method = "circle")


#decision tree learning
library(rpart)
library(rpart.plot)
colnames(testdata10) <- colnames(trainingdata10h)

tree.5000 <- rpart(classLabel~.,data=trainingdata10h[c(-2,-3,-4)],control=rpart.control(minsplit=5000,cp=0))
tree.1000 <- rpart(classLabel~.,data=trainingdata10h[c(-2,-3,-4)],control=rpart.control(minsplit=1000,cp=0))
tree.300 <- rpart(classLabel~.,data=trainingdata10h[c(-2,-3,-4)],control=rpart.control(minsplit=300,cp=0))
tree.10 <- rpart(classLabel~.,data=trainingdata10h[c(-2,-3,-4)],control=rpart.control(minsplit=10,cp=0))
tree.2 <- rpart(classLabel~.,data=trainingdata10h[c(-2,-3,-4)],control=rpart.control(minsplit=2,cp=0))

prp(tree.5)


tree.2.train= sum(as.character(predict(tree.2, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10[,42])/length(trainingdata10[,42])
tree.2.test= sum(as.character(predict(tree.2, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

tree.10.train= sum(as.character(predict(tree.10, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10[,42])/length(trainingdata10[,42])
tree.10.test= sum(as.character(predict(tree.10, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

tree.300.train= sum(as.character(predict(tree.300, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10[,42])/length(trainingdata10[,42])
tree.300.test= sum(as.character(predict(tree.300, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

tree.1000.train= sum(as.character(predict(tree.1000, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10[,42])/length(trainingdata10[,42])
tree.1000.test= sum(as.character(predict(tree.1000, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

tree.5000.train= sum(as.character(predict(tree.5000, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10[,42])/length(trainingdata10[,42])
tree.5000.test= sum(as.character(predict(tree.5000, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

#randomForest

library("randomForest")
tuRF= tuneRF(trainingdata10h[c(-2,-3,-4,-42)],trainingdata10h$classLabel)

randForest <- randomForest(classLabel~.,data=trainingdata10h[c(-2,-3,-4)], ntree=4, mtry = 24)

randtree.4.train= sum(as.character(predict(randForest, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10h[,42])/length(trainingdata10h[,42])
randtree.4.test= sum(as.character(predict(randForest, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

randForest <- randomForest(classLabel~.,data=trainingdata10h[c(-2,-3,-4)], ntree=20, mtry = 24)

randtree.20.train= sum(as.character(predict(randForest, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10h[,42])/length(trainingdata10h[,42])
randtree.20.test= sum(as.character(predict(randForest, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

randForest <- randomForest(classLabel~.,data=trainingdata10h[c(-2,-3,-4)], ntree=40, mtry = 24)

randtree.40.train= sum(as.character(predict(randForest, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10h[,42])/length(trainingdata10h[,42])
randtree.40.test= sum(as.character(predict(randForest, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

randForest <- randomForest(classLabel~.,data=trainingdata10h[c(-2,-3,-4)], ntree=200, mtry = 24)

randtree.200.train= sum(as.character(predict(randForest, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10h[,42])/length(trainingdata10h[,42])
randtree.200.test= sum(as.character(predict(randForest, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

randForest <- randomForest(classLabel~.,data=trainingdata10h[c(-2,-3,-4)], ntree=500, mtry = 24)

randtree.500.train= sum(as.character(predict(randForest, trainingdata10h[c(-2,-3,-4)], type = c("class")))==trainingdata10h[,42])/length(trainingdata10h[,42])
randtree.500.test= sum(as.character(predict(randForest, testdata10[c(-2,-3,-4)], type = c("class")))==testdata10[,42])/length(testdata10[,42])

# paralllel  random forest
library("foreach")
library("randomForest")
colnames(testdata10) <- colnames(trainingdata10h)
tic = proc.time();
prf <- foreach(ntree=rep(1, 4), .combine=combine) %dopar% randomForest(as.matrix(trainingdata10[c(-2,-3,-4,-42)]), trainingdata10$classLabel, ntree=ntree)
toc = proc.time();
prf4= toc-tic
show(toc-tic)
tic = proc.time();

prf <- foreach(ntree=rep(5, 4), .combine=combine) %dopar% randomForest(as.matrix(trainingdata10[c(-2,-3,-4,-42)]), trainingdata10$classLabel, ntree=ntree)
toc = proc.time();
prf20= toc-tic
show(toc-tic)


tic = proc.time();
prf <- foreach(ntree=rep(10, 4), .combine=combine) %dopar% randomForest(as.matrix(trainingdata10[c(-2,-3,-4,-42)]), trainingdata10$classLabel, ntree=ntree)
toc = proc.time();
prf40= toc-tic

tic = proc.time();
prf <- foreach(ntree=rep(50, 4), .combine=combine) %dopar% randomForest(as.matrix(trainingdata10[c(-2,-3,-4,-42)]), trainingdata10$classLabel, ntree=ntree)
toc = proc.time();
prf200= toc-tic
show(toc-tic)

tic = proc.time();
prf <- foreach(ntree=rep(125, 4), .combine=combine) %dopar% randomForest(as.matrix(trainingdata10[c(-2,-3,-4,-42)]), trainingdata10$classLabel, ntree=ntree)
toc = proc.time();
prf500= toc-tic
show(toc-tic)



#multinomial logistic regression
library("glmnet")

glm1000 = glmnet(as.matrix(trainingdata10h[c(-2,-3,-4,-42)]),trainingdata10h$classLabel, family="multinomial",maxit=100)
glm1000predMod = predict(glm1000, type = c("class"),newx= as.matrix(trainingdata10h[c(-2,-3,-4,-42)]))
glm1000.train= sum(as.character(glm1000predMod[,dim(glm1000predMod)[2]])==trainingdata10h[,42])/length(trainingdata10h[,42])
glm1000predMod = predict(glm1000, type = c("class"),newx= as.matrix(testdata10[c(-2,-3,-4,-42)]))
glm1000.test =  sum(as.character(glm1000predMod[,dim(glm1000predMod)[2]])==testdata10[,42])/length(testdata10[,42])



atree.5 <- rpart(classLabel~.,data=trainingdata10h[c(5,8,9,10,20,31,42)],control=rpart.control(minsplit=5000,cp=0))
btree.5 <- rpart(classLabel~.,data=trainingdata10h[c(11,6,7,15,25,36,42)],control=rpart.control(minsplit=5000,cp=0))
ctree.5 <- rpart(classLabel~.,data=trainingdata10h[c(12,13,14,17,23,24,27,28,29,42)],control=rpart.control(minsplit=3000,cp=0))
prp(atree.5)
prp(btree.5)
prp(ctree.5)

