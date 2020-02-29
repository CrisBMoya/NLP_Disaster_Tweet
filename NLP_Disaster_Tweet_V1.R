# Load libraries
library(tidyverse)
library(h2o)
library(caret)
library(dummies)
library(plyr)

# Set working directory
setwd(gsub(pattern='Documents', replacement='Google Drive/Github/NLP_Disaster_Tweet', x=getwd()))

# Load Data
DataDF=read_delim(file=unz(description='Data/nlp-getting-started.zip', filename='train.csv'), delim=',')

# Dummies
Dummy=dummy.data.frame(data=as.data.frame(DataDF[,c('keyword')]), dummy.classes='ALL')
Dummy=bind_cols(Dummy, DataDF[,c('text','target')])

# Split data into dev and train sets
Split=createDataPartition(y=DataDF$target, times=1, p=0.8)
Split=Split$Resample1

TrainDF=Dummy[Split,]
DevDF=Dummy[-Split,]

# Start H2O
h2o.init(nthreads=-1, ip='localhost', port=8079)

# Import Data
TrainDF_H2O=as.h2o(TrainDF)
DevDF_H2O=as.h2o(DevDF)

# Model
GBM_Baseline_Model=h2o.gbm(x=setdiff(x=colnames(TrainDF), y=c('text','target','id')), y='target', training_frame=TrainDF_H2O, validation_frame=DevDF_H2O)

# Performance
Performance_Baseline=h2o.performance(model=GBM_Baseline_Model, newdata=DevDF_H2O)

# Predict
Predict_Baseline=h2o.predict(GBM_Baseline_Model, DevDF_H2O)
Pred_Baseline=ifelse(test=as.data.frame(Predict_Baseline)[,1]>=0.5, yes=1, no=0)
confusionMatrix(as.factor(Pred_Baseline), as.factor(DevDF$target))
h2o.varimp_plot(model=GBM_Baseline_Model)

# NLP

# Stop Words
StopWords=readLines(con='Data/StopWords.txt', warn=FALSE)

# Tokenize
Token=function(x, trailling){
  RawToken=unlist(strsplit(x=x, split=' '))
  RawToken=RawToken[!(RawToken %in% StopWords)]
  RawToken=gsub(pattern='[[:punct:]]|[[:blank:]]', replacement='', x=RawToken)
  RawToken=tolower(RawToken)
  
  if(trailling==TRUE){
    return(c(RawToken,NA))
  }else{
    return(RawToken)
  }
}
OriginalTokens=sapply(X=Dummy$text, FUN=function(x){Token(x=x, trailling=FALSE)})
Tokens=unlist(OriginalTokens)

# To DF
Tokens=Tokens[-which(sapply(X=Tokens, FUN=nchar)==0)]
TokenDF=as.data.frame(x=Tokens, stringsAsFactors=FALSE)

# To H2O
TokenDF_H2O=as.character(as.h2o(TokenDF$Tokens))

# Transform
WordVector=h2o.word2vec(training_frame=TokenDF_H2O, vec_size=100)
h2o.findSynonyms(word2vec=WordVector, word='disaster')

# Word2Vec
VectorToken=sapply(X=Dummy$text, FUN=function(x){Token(x=x, trailling=TRUE)})
VectorToken=unlist(VectorToken)
names(VectorToken)=NULL

# As h2o
TokenDFVec_H2O=as.character(as.h2o(VectorToken))

# Transform
WordVecTrans=h2o.transform_word2vec(word2vec=WordVector, words=TokenDFVec_H2O, aggregate_method='AVERAGE')
dim(WordVecTrans)

# Cbind
Extra_DF=bind_cols(Dummy, as.data.frame(WordVecTrans))

# Split
Extra_TrainDF=Extra_DF[Split,]
Extra_DevDF=Extra_DF[-Split,]

# To H2O
Extra_TrainDF_H2O=as.h2o(Extra_TrainDF)
Extra_DevDF_H2O=as.h2o(Extra_DevDF)

# New Model
GBM_Model=h2o.gbm(x=setdiff(x=colnames(Extra_TrainDF), y=c('text','target','id')), y='target', training_frame=Extra_TrainDF_H2O, validation_frame=Extra_DevDF_H2O)

# New Model Performance
Performance_Model=h2o.performance(model=GBM_Model, newdata=Extra_DevDF_H2O)

# New Model Predict
Predict_Model=h2o.predict(GBM_Model, Extra_DevDF_H2O)
Pred_Model=ifelse(test=as.data.frame(Predict_Model)[,1]>=0.5, yes=1, no=0)
confusionMatrix(as.factor(Pred_Model), as.factor(Extra_DevDF$target))
h2o.varimp_plot(model=GBM_Model)

#############
#############

# Predict New Data
TestDF=read_delim(file=unz(description='Data/nlp-getting-started.zip', filename='test.csv'), delim=',')

# Dummies
Test_Dummy=dummy.data.frame(data=as.data.frame(TestDF[,c('keyword')]), dummy.classes='ALL')
Test_Dummy=bind_cols(Test_Dummy, TestDF[,c('text')])



Test_OriginalTokens=sapply(X=Test_Dummy$text, FUN=function(x){Token(x=x, trailling=FALSE)})
Test_Tokens=unlist(Test_OriginalTokens)

# To DF
Test_Tokens=Test_Tokens[-which(sapply(X=Test_Tokens, FUN=nchar)==0)]
Test_TokenDF=as.data.frame(x=Test_Tokens, stringsAsFactors=FALSE)

# To H2O
Test_TokenDF_H2O=as.character(as.h2o(Test_TokenDF$Test_Tokens))

# Transform
Test_WordVector=h2o.word2vec(training_frame=Test_TokenDF_H2O, vec_size=100)

# Word2Vec
Test_VectorToken=sapply(X=Test_Dummy$text, FUN=function(x){Token(x=x, trailling=TRUE)})
Test_VectorToken=unlist(Test_VectorToken)
names(Test_VectorToken)=NULL

# As h2o
Test_TokenDFVec_H2O=as.character(as.h2o(Test_VectorToken))

# Transform
Test_WordVecTrans=h2o.transform_word2vec(word2vec=Test_WordVector, words=Test_TokenDFVec_H2O, aggregate_method='AVERAGE')
dim(Test_WordVecTrans)

# Cbind
Test_Extra_DF=bind_cols(Test_Dummy, as.data.frame(Test_WordVecTrans))
Test_Extra_DF_H2O=as.h2o(Test_Extra_DF)

# Prediction on Test Set
Test_Predict_Model=h2o.predict(GBM_Model, Test_Extra_DF_H2O)
Test_Pred_Model=ifelse(test=as.data.frame(Test_Predict_Model)[,1]>=0.5, yes=1, no=0)

# Format Submission
Submission_v1=data.frame('id'=TestDF$id, 'target'=Test_Pred_Model)
write_delim(x=Submission_v1, path='NLP_v1.csv', delim=',')

#Submit
RE=TRUE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c nlp-getting-started -f NLP_v1.csv -m "Submission from API - NLP_v1"')
}

# Shutdown H2O
h2o.shutdown(prompt=FALSE)
