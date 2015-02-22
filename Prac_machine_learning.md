---
title: "Practical Machine Learning"
output: html_document
---

#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

#Loading the training and test data

The training and the test data were first loaded into R. 


```r
train_data=read.csv("pml-training.csv", h=T, stringsAsFactors=F)
test_data=read.csv("pml-testing.csv", h=T,stringsAsFactors=F)
dim(train_data)
```

```
## [1] 19622   160
```

The training data had 19622 rows and 160 columns and the test data had 20 rows and 160 columns. 

##Removing columns with NA's and missing values

There are several columns in the dataset with NA's and missing values. We first remove any columns, which has more than 95% NA's in them.


```r
na_factor<-apply(train_data,2,function(col)sum(is.na(col))/length(col))*100
na_factor=as.data.frame(na_factor)
na_factor$colname=rownames(na_factor)
rownames(na_factor)=NULL
#plot_na=ggplot(subset(na_factor,na_factor>95), aes(colname, na_factor))+geom_bar(stat="identity")+coord_flip()
#plot_na
subset_na_95=subset(na_factor, na_factor>95)
na_thrash <- !names(train_data) %in% subset_na_95$colname
train_data_trans <- train_data[,na_thrash] 
```

## missing values


```r
still<-c("kurtosis_roll_belt",  "kurtosis_picth_belt",  "kurtosis_yaw_belt",	"skewness_roll_belt",	"skewness_roll_belt.1",	"skewness_yaw_belt", "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "kurtosis_roll_arm",	"kurtosis_picth_arm",	"kurtosis_yaw_arm",	"skewness_roll_arm",	"skewness_pitch_arm",	"skewness_yaw_arm", "kurtosis_roll_arm",	"kurtosis_picth_arm",	"kurtosis_yaw_arm",	"skewness_roll_arm",	"skewness_pitch_arm",	"skewness_yaw_arm", "kurtosis_roll_dumbbell",	"kurtosis_picth_dumbbell",	"kurtosis_yaw_dumbbell",	"skewness_roll_dumbbell",	"skewness_pitch_dumbbell",	"skewness_yaw_dumbbell" , "max_yaw_dumbbell" ,"min_yaw_dumbbell" ,"amplitude_yaw_dumbbell", "max_yaw_forearm", "max_yaw_forearm", "min_yaw_forearm","kurtosis_picth_forearm", "skewness_yaw_forearm","amplitude_yaw_forearm","kurtosis_picth_forearm","kurtosis_yaw_forearm","skewness_roll_forearm","kurtosis_roll_forearm","skewness_pitch_forearm")
still_thrash <- !names(train_data_trans) %in% still
train_data_trans_1 <- train_data_trans[,still_thrash] 
train_data_trans_1<- train_data_trans_1[,-c(1:7)]
train_data_trans_1$classe=as.factor(train_data$classe)
```

## Spliting the training data

We will first split the training data into training and validation dataset. The models built using the training dataset will be validated using the validation dataset.



```r
set.seed(77777)
library(caret)

training <- createDataPartition(y=train_data_trans_1$classe,
                               p=0.75,list=FALSE)
training_data <- train_data_trans_1[training,]
validation_data <- train_data_trans_1[-training,]
```




#Data Preprocessing 

We first look for correlation between the predictors using the cor function and plot the correlation coefficients using the heatmap function. 



```r
m <- abs(cor(training_data[,-53]))
diag(m) <-0
which(m>0.8, arr.ind=T)
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_y       9   1
## accel_belt_z      10   1
## accel_belt_x       8   2
## magnet_belt_x     11   2
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_y       9   4
## accel_belt_z      10   4
## pitch_belt         2   8
## magnet_belt_x     11   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_z      10   9
## roll_belt          1  10
## total_accel_belt   4  10
## accel_belt_y       9  10
## pitch_belt         2  11
## accel_belt_x       8  11
## gyros_arm_y       19  18
## gyros_arm_x       18  19
## magnet_arm_x      24  21
## accel_arm_x       21  24
## magnet_arm_z      26  25
## magnet_arm_y      25  26
## accel_dumbbell_x  34  28
## accel_dumbbell_z  36  29
## gyros_dumbbell_z  33  31
## gyros_forearm_z   46  31
## gyros_dumbbell_x  31  33
## gyros_forearm_z   46  33
## pitch_dumbbell    28  34
## yaw_dumbbell      29  36
## gyros_forearm_z   46  45
## gyros_dumbbell_x  31  46
## gyros_dumbbell_z  33  46
## gyros_forearm_y   45  46
```

```r
plot(training_data[,21], training_data[,24])
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 

```r
heatmap(m)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-2.png) 

We see that 38 of the variables have a high correlation coefficient greater than 0.8. We use principal component analysis (PCA) to preprocess the data. The variance threshold was set to 0.95 meaning we retain all the PC's that explain 95% of the variance in the training data.



```r
preProc<- preProcess (training_data[,-53], method="pca", thresh=0.95)
training_data_preprocess<- predict (preProc, training_data[,-53])
training_data_preprocess$classe=training_data$classe

validation_data_preprocess <- predict (preProc, validation_data[,-53])
validation_data_preprocess$classe=validation_data$classe
```

# Model Building using Random forest

The popular machine learnign method namely random forest was first selected to fit the training data. Since transformation of the data doesn't make a big difference with tree methods, we try to fit a randomforest model to the original unprocessed training data. To see if the preprocessing had an effect, we also fit a randomforest model to the preprocessed data and compare them.

## The out-of-bag (oob) error estimate
In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run, as follows:

Each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.

Put each case left out in the construction of the kth tree down the kth tree to get a classification. In this way, a test set classification is obtained for each case in about one-third of the trees. At the end of the run, take j to be the class that got most of the votes every time case n was oob. The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. This has proven to be unbiased in many tests.

### randomForest model without preprocessing


```r
library(randomForest)
fit<-randomForest(classe~., data=training_data)
print(fit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training_data) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.44%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4184    0    0    1    0 0.0002389486
## B   15 2830    3    0    0 0.0063202247
## C    0   12 2553    2    0 0.0054538372
## D    0    0   24 2387    1 0.0103648425
## E    0    0    1    6 2699 0.0025868441
```

```r
confusionMatrix(table(predict(fit, training_data), training_data$classe))
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 4185    0    0    0    0
##   B    0 2848    0    0    0
##   C    0    0 2567    0    0
##   D    0    0    0 2412    0
##   E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
#validation_data_preprocess<- predict(preProc, validation_data)
confusionMatrix(table(predict(fit, validation_data), validation_data$classe))
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 1393    5    0    0    0
##   B    1  942    6    0    0
##   C    1    2  849    8    0
##   D    0    0    0  795    2
##   E    0    0    0    1  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9922, 0.9965)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9926   0.9930   0.9888   0.9978
## Specificity            0.9986   0.9982   0.9973   0.9995   0.9998
## Pos Pred Value         0.9964   0.9926   0.9872   0.9975   0.9989
## Neg Pred Value         0.9994   0.9982   0.9985   0.9978   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1921   0.1731   0.1621   0.1833
## Detection Prevalence   0.2851   0.1935   0.1754   0.1625   0.1835
## Balanced Accuracy      0.9986   0.9954   0.9951   0.9942   0.9988
```

The training accuracy from the above model is 100% and the OOB estimate of the error rate is 0.4%, which is excellent and the validation accuracy is 99.5%.

### randomForest model with pca preprocessing


```r
fit_preprocess<-randomForest(classe~., data=training_data_preprocess)
print(fit_preprocess)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training_data_preprocess) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 2.36%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4152    9   12   10    2 0.007885305
## B   48 2752   43    2    3 0.033707865
## C    2   27 2499   34    5 0.026490066
## D    3    2   92 2309    6 0.042703151
## E    2   12   21   12 2659 0.017368810
```

```r
confusionMatrix(table(predict(fit_preprocess, training_data_preprocess), training_data_preprocess$classe))
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 4185    0    0    0    0
##   B    0 2848    0    0    0
##   C    0    0 2567    0    0
##   D    0    0    0 2412    0
##   E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(table(predict(fit_preprocess, validation_data_preprocess), validation_data_preprocess$classe))
```

```
## Confusion Matrix and Statistics
## 
##    
##        A    B    C    D    E
##   A 1388   15    1    2    0
##   B    1  923   13    1    7
##   C    4   11  826   30    5
##   D    2    0   13  771    4
##   E    0    0    2    0  885
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9774          
##                  95% CI : (0.9728, 0.9813)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9714          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9950   0.9726   0.9661   0.9590   0.9822
## Specificity            0.9949   0.9944   0.9877   0.9954   0.9995
## Pos Pred Value         0.9872   0.9767   0.9429   0.9759   0.9977
## Neg Pred Value         0.9980   0.9934   0.9928   0.9920   0.9960
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2830   0.1882   0.1684   0.1572   0.1805
## Detection Prevalence   0.2867   0.1927   0.1786   0.1611   0.1809
## Balanced Accuracy      0.9949   0.9835   0.9769   0.9772   0.9909
```

The training accuracy from the above model is 100% and the OOB estimate of the error rate is 2.3%, which is excellent and the validation accuracy is 97%.

# Model Selection and test set prediction

Since the validation accuracy was higher in the random forest model with the original non-preprocessed training data, we chose that to be the final model.
The model was then used to predict the classes of the test dataset.


```r
test_data=read.csv("pml-testing.csv", h=T,stringsAsFactors=F)
train_lables<- colnames(test_data) %in% colnames(training_data)
test_data_filtered<- test_data[train_lables]
setdiff(names(training_data), names(test_data_filtered))
```

```
## [1] "classe"
```

```r
pred_test<-predict(fit, test_data_filtered)
pred_test
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

# Writing the results

The pml_write_files function from the course site was used to write the results of the test dataset and they were then submitted to the course site for evaluation.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers<-as.character(pred_test)
pml_write_files(answers)
```

#References

 1. http://groupware.les.inf.puc-rio.br/har
 
 2. https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
