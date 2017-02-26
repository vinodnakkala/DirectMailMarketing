install.packages("dplyr")
install.packages("corrplot")
install.packages("mice")
install.packages("dummies")
## Install and load the packages required to this Data Challenge
library(caret)
require(data.table)
library(dplyr)
library(h2o)
library(glmnet)
library(corrplot)
library(mice)
library(dummies)
library(pROC)
library(xgboost)

# Set the workign Directory in which the CSV files are present
setwd('C:/Users/nakka/Downloads/SparkCognitionDataScienceAssignment/SparkCognitionDataScienceAssignment')
# Read the Training and test CSV files
train <- fread('marketing_training.csv',header=T,na.strings = "",stringsAsFactors = T)
test <- fread('marketing_test.csv',header=T,na.strings = "",stringsAsFactors = T)
# Observe the summary of training dataset
dim(train)
summary(train)
#View(train)
## IMpute Minning Level of schoolign with new level "Not Given" in both training and test dataset
train[is.na(train$schooling),"schooling"]="Not Given"
test[is.na(test$schooling),"schooling"]="Not Given"
levels(train$schooling)
# Create a new label for observations whose age is Missing, indicating whether age is missing or present in dataset
train$IsAgeMissing = ifelse(is.na(train$custAge),"1","0")
train$IsAgeMissing = as.factor(train$IsAgeMissing)
test$IsAgeMissing = ifelse(is.na(test$custAge),"1","0")
test$IsAgeMissing = as.factor(test$IsAgeMissing)
# Observe the variable values and correlation
table(train$pdays)
table(train$pmonths)
class(train$poutcome)
cor(train$pmonths,train$pdays)
# Can remove one of the Variables as correlation is almost 1 or can use techniques while modelling to address this
# Observe if Age is depending on any of these variables
plot(train$custAge,train$profession)
plot(train$custAge,train$marital)
plot(train$custAge,train$schooling)
plot(train$custAge,train$default)
plot(train$custAge,train$housing)
plot(train$custAge,train$loan)
plot(train$custAge,train$contact)
# Age seems to be related to these variables
train$custAge = as.numeric(train$custAge)
test$custAge = as.numeric(test$custAge)
# Impute Age using MICE "Multivariate Imputation by Chained Equations" One of the effective ways to do than
# Using Mean/Median
train = complete(mice(train))
test = complete(mice(test))
# Find correlation of all continuous variables and observe using corrplot
trainCont = train[,c("custAge","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","pdays","pmonths","nr.employed")]
M <- cor(trainCont)
corrplot(M, method="circle")
# Store responses in a temporary variable
Outcome = train$responded
train$responded = NULL
train$responded = Outcome
# Label the target variable as 0/1 instead of yes/no and as factor
train$responded = ifelse(train$responded == 'yes',1,0)
train$responded = as.factor(train$responded)
# Store the ID of each observation from test set and remove from the data frame
ID = test$V1
test$V1 = NULL
# Make sure that dimensions of train and test are same, except the target variable
dim(train)
dim(test)
summary(train$responded)
# Get all the independant and depandant variables in temporary variables
ind.var=c("custAge","profession","marital","schooling","default","housing","loan","contact","month",
          "day_of_week","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx",
          "euribor3m","nr.employed","pmonths","pastEmail","IsAgeMissing")
dep.var="responded"
##############################################################################################
#impactCoding = function(TrainDf,TestDf,ImpactVariable,TargetVariable,ImpactVarName)
#{
#  tout=tapply(TargetVariable,ImpactVariable,mean)
#  dfout=as.data.frame(tout)
#  dfout$levels=rownames(dfout)
#  rownames(dfout)=NULL
#  UpdatedTrained = merge(x=TrainDf,y=dfout, by.x = ImpactVarName,by.y = "levels", all.x=T)
#  colnames(UpdatedTrained)[colnames(UpdatedTrained)=="tout"]= paste(ImpactVarName,"_impact")
#  UpdatedTest = merge(x=TestDf,y=dfout, by.x = ImpactVarName,by.y = "levels", all.x=T)
#  colnames(UpdatedTest)[colnames(UpdatedTest)=="tout"]= paste(ImpactVarName,"_impact")
#  list(UpdatedTrained,UpdatedTest)
#}
#temp1=impactCoding(train,test,train$profession,train$poutput,"profession")
#str(temp1)
#Trained1 = temp1[[1]]
#View(Trained1)
#Test1 = temp1[[2]]
#View(Test1)
#colnames(temp1)[colnames(temp1)=="tout"]
#View(temp1)
#class(train1)
#train$profession
###################################################################################################
####################################################################################################
# Use H2o package for building Random Forest with 5 fold cross Validation
# Allocate 4 GB memory for H2o
# This will take HUGE Runtime ..!!
localh2o = h2o.init(nthreads = -1, max_mem_size = "4G")
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)
# Fit Random Forest model and tune the parameters based on Cross Validation Scores
rf_fit = h2o.randomForest(x = ind.var,
                          y = dep.var,
                          training_frame = train_h2o,
                          model_id = "rf_fit",
                          ntrees = 300,
                          keep_cross_validation_predictions = T,
                          score_each_iteration = T,
                          seed = 1000,
                          stopping_metric = "AUC",
                          nfolds=5)
# Observe the Summary of Random Forest model and predict on Test set
summary(rf_fit)
rf_Predict <- h2o.predict(object=rf_fit,newdata = test_h2o)
head(rf_Predict)
########################## Logistic Regression with Ridge  #############################
## For Logistic model with Regularization prepare data, by creating Dummy Variables
summary(train)
dum = dummyVars(~profession+marital+schooling+default+housing+loan+contact+month+day_of_week+
                 poutcome,data=train)
dummy=cbind(train[c("custAge","campaign","pdays","previous","emp.var.rate","cons.price.idx",
                    "cons.conf.idx","euribor3m","nr.employed","pmonths","pastEmail","IsAgeMissing",
                    "responded")],
            head(predict(dum,train),
                 n=nrow(train)))
# Convert all the variables to Numeric as the input to glmnet is Matrix
dummy$IsAgeMissing=as.numeric(as.character(dummy$IsAgeMissing))
dummy=as.matrix(dummy)
dummy = as(dummy, "dgCMatrix")
indp.var=setdiff(colnames(dummy),(dep.var))
# Fit model using Ridge regression 
glmnetModel <- cv.glmnet(x=dummy[,indp.var],y=dummy[,dep.var], alpha = 0, family = "binomial", type.measure = "auc")
# Observe the summary of model with Lambda and CV scores
summary(glmnetModel)
str(glmnetModel)
# Predict using glmnet on training data to see how probabilities are given
# This can be used to decide Probability cutoff later
glmnetPredict1 <- predict(glmnetModel, dummy[,indp.var], s="lambda.min",type="response")
# Calculate and see ROC
roc_obj <- roc(dummy[,"responded"], glmnetPredict1)
auc(roc_obj)
# Prepare test data similar to training data to get predictions using glmnet
dum_test = dummyVars(~profession+marital+schooling+default+housing+loan+contact+month+day_of_week+
                  poutcome,data=test)
dummy_test=cbind(test[c("custAge","campaign","pdays","previous","emp.var.rate","cons.price.idx",
                    "cons.conf.idx","euribor3m","nr.employed","pmonths","pastEmail","IsAgeMissing")],
            head(predict(dum_test,test),
                 n=nrow(test)))
dummy_test$IsAgeMissing=as.numeric(as.character(dummy_test$IsAgeMissing))
# Create Dummy Variables to get the matix in test also having same dimension as training
dummy_test$default.yes = 0
dummy_test$schooling.illiterate = 0
dummy_test=as.matrix(dummy_test)
dummy_test = as(dummy_test, "dgCMatrix")
# Predict on test data using the model being created
glmnetPredict <- predict(glmnetModel, dummy_test[,indp.var], s="lambda.min",type="response")
head(glmnetPredict)
######################################## Xtreme Gradient Boosting ##############################
## Now lets try xgboost and set the parametrs, these need to be tuned
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",  # maximizing for auc
                eta                 = 0.002,   # learning rate - Number of Trees
                max_depth           = 7,      # maximum depth of a tree
                subsample           = .9,     # subsample ratio of the training instance
                colsample_bytree    = .87,    # subsample ratio of columns 
                min_child_weight    = 1,      # minimum sum of instance weight (defualt)
                scale_pos_weight    = 1       # helps convergance bc dataset is unbalanced
)
train$responded=as.numeric(as.character(train$responded))
# Convert dataset to sparse matrix to send to xgboost
train_new <- sparse.model.matrix(train$responded ~ ., data = train)
dtrain <- xgb.DMatrix(data=train_new, label=train$responded)
# Fit the model
model_xgb <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 50, 
                          verbose             = 1,
                          maximize            = FALSE
)
# Score on test data by converting it to Sparse Matrix
test$responded <- -1
testing <- sparse.model.matrix(responded ~ ., data = test)
preds <- predict(model_xgb, testing)
test$pred_xgb <- preds
test$target <- NULL
head(test$pred_xgb)
## Observe the probabilities of Random Forest, GLMNET and XGBOOST

#### Adjusting Probabilities ############
## GlmNet has Highest AUC, lets try to find Probability cutoff to maximize F1 Score
train$glmpredict_prob = glmnetPredict1
train$glmpredict_class = ifelse(train$glmpredict_prob < 0.23,0,1)
table(train$responded,train$glmpredict_class)
prec=nrow(train[train$responded == 1 & train$glmpredict_class == 1,])/nrow(train[train$glmpredict_class == 1,])
rec = nrow(train[train$responded == 1 & train$glmpredict_class == 1,])/nrow(train[train$responded == 1,])
2*prec*rec/(prec+rec)
# At probability cutoff of 0.23 we get Maximum F1 Score
############################################
as.vector(glmnetPredict)
test$pred_glmnet = glmnetPredict
# Label the class based on probability cutoff obtained
test$glmnet_class = ifelse(test$pred_glmnet<0.23,0,1)
summary(test)
# Prepare submission File with ID and class label predicted
Submit = cbind(ID,test$glmnet_class)
as.data.frame(Submit)
colnames(Submit)
# Write submission file to disk
colnames(Submit)=c("ID","Prediction")
write.csv(Submit,"submission.csv",row.names=FALSE)

########################################################################################








