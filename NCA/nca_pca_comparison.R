library(e1071) #Library for SVM
library(class) #Library for KNN

####################################################################################################
# Helper function for mean adjustment of dataset - also used as one initialization technique for A #
####################################################################################################

mean_adj <- function(x,d=ncol(x),D=ncol(x))
{
  x_min <- apply(x, 2, min)
  x <- sweep(x, 1, x_min)
  x_max <- apply(x, 2, max)
  #print(x_max)
  x<-sweep(x,1,x_max-x_min,'/')
  
  diag(1/(x_max),d,D)
}


#######################################################
# Helper function for printing classification results #
#######################################################

classification_report <- function(t, c=0)
{
  if(c==0) { cat("\nNCA Results:-\n")}
  else { cat("\nPCA Results:-\n") }
  acc<-(t[1,1]+t[2,2])/(t[1,1]+t[2,2]+t[1,2]+t[2,1])
  pre<-t[2,2]/(t[2,2]+t[2,1])
  rec<-t[2,2]/(t[2,2]+t[1,2])
  f1<-2*pre*rec/(pre+rec)
  cat(t,"\n\nAccuracy = ",acc,"\nPrecision = ",pre,"\nRecall = ",rec,"\nF1 Score = ",f1)
}

##################################################################################################
# Function to give NCA transformation matrix - accuracy limited by initial matrix and iterations #
##################################################################################################

nca <- function(x, y, A_init = diag(ncol(x)), iters=1e2, lr = 0.01)
{
  x <- as.matrix(x)
  #y <- as.factor(y) #Needed if string classes - not encoded
  
  A <- A_init
  
  N <- nrow(x)
  stopifnot(NROW(x) == length(y))
  
  p <- numeric(N)
  p_tot <- numeric(iters)
  for (it in seq_len(iters)){
    for (i in seq_len(N)){

      Diff <- tcrossprod(A, x)      
      Diff <- (Diff - as.numeric(Diff[,i]))
      p_ik <- exp(-colSums(Diff*Diff))     #Softmax  
      
      p_ik[i] <- 0 #Leave one out
      softmax <- sum(p_ik)
      if (softmax > .Machine$double.eps){ #Smallest float value
        p_ik <- p_ik/sum(p_ik)             
      }
      
      
      #Correct labelling using chosen neighbors at random
      true_y <- y == y[i] 
      
      p[i] <- sum(p_ik[true_y]) #Probability ith point is classified correctly by knn using random neighbors
      
      diff    <- t(t(x) - as.numeric(x[i,]))  
      pdiff <- p_ik * diff                    
      
      #Gradient descent
      grad <- (p[i]*crossprod(diff, pdiff)) - crossprod(diff[true_y,], pdiff[true_y,]) 
      A <- A + lr * (A %*% grad) 

    }
    p_tot[it] <- sum(p)
  }
  
  list(A = A, p_tot=p_tot, p = p, A_adj = A/A[1,1])
}


##########################################################################################################
#                                             PROGRAM START!!!!                                          #
##########################################################################################################

diabetes<-read.csv(file="C:\\Users\\Me\\Downloads\\pima-indians-diabetes-database\\diabetes.csv")

X=diabetes[,1:8]
Y=diabetes[,9]

#75% of the dataset
smp_size <- floor(0.75 * nrow(diabetes))

#Seed to make partition reproducible - COMMENT OUT FOR RANDOMIZATION
set.seed(0)

train_ind <- sample(seq_len(nrow(diabetes)), size = smp_size)

train <- diabetes[train_ind, ]
test <- diabetes[-train_ind, ]

Xtrain<-train[,1:8]
Ytrain<-train[,9]
Xtest<-test[,1:8]
Ytest<-test[,9]

###NCA variables
x<-Xtrain
xtest<-Xtest

x <- as.matrix(x)
xtest<-as.matrix(xtest)

labels<-Ytrain

###PCA variables
x2<-Xtrain
xtest2<-Xtest

x2 <- as.matrix(x2)
xtest2<-as.matrix(xtest2)

labels2<-Ytrain

#No. of attributes of dataset
D=ncol(x)

### CHANGE FINAL NO. OF DIMENSIONS HERE ###

#No. of dimensions to reduce to
d=4


####################################################
#                 NCA dim reduction                #
####################################################

#A<-diag(1,d,D,FALSE) #CHOOSE FOR IDENTITY INITIALIZATION
A<-replicate(D,rnorm(d)) #CHOOSE FOR RANDOM INITIALIZATION
#A <- mean_adj(x,d,D) #CHOOSE FOR MEAN ADJUSTED DIAGONAL MATRIX AS A, ELSE UNCOMMENT THE FOLLOWING LINE
mean_adj(x,d,D) #COMMENT OUT IF THIRD CASE CHOSEN, ELSE KEEP



results <- nca(x=x, y = labels, A_init = A, iters = 1000, lr = 1e-2)
#results$A_adj


Xnca_train<-t(tcrossprod(res$A, x))
Xnca_test<-t(tcrossprod(res$A, xtest))

#colus<-c("red","blue")
#plot(Xnca_train,col=colus[diabetes$Outcome+1])



######################################################
#                   PCA dim reduction                #
######################################################

pca<-prcomp(x2)
Apca<-t(pca$rotation[,1:d])

Xpca_train<-t(tcrossprod(Apca, x2))
Xpca_test<-t(tcrossprod(Apca, xtest2))

#colus2<-c("red","blue")
#plot(Xpca_train,col=colus2[diabetes$Outcome+1])



######################################################
#                 Classification                     #
######################################################

########################################################################
#### SVM - Not guaranteed to improve by either preprocessing algorithm #
########################################################################

cat("\n***SVM RESULTS***\n")


ncaclassifier = svm(formula = labels~., data = Xnca_train, type = 'C-classification', 
                 kernel = 'linear') 
ynca_pred = predict(ncaclassifier, newdata = Xnca_test)
t1<-table(ynca_pred,Ytest)
classification_report(t1,0)


pcaclassifier = svm(formula = labels2~., data = Xpca_train, type = 'C-classification', 
                    kernel = 'linear') 
ypca_pred = predict(pcaclassifier, newdata = Xpca_test)
t2<-table(ypca_pred,Ytest)
classification_report(t2,1)


########################################################################################
#### With KNN (metric based classification) - should be better using NCA preprocessing #
########################################################################################

cat("\n***KNN RESULTS***\n")


K=sqrt(smp_size)+1 #Kept as sqrt of sample size and odd. Can be modified to explore other values.
#Other means to choose K - cross validation, trial and error 


ynca_pred2=knn(Xnca_train,Xnca_test,cl=labels,k=K)
t3<-table(ynca_pred2,Ytest)
classification_report(t3,0)


ypca_pred2=knn(Xpca_train,Xpca_test,cl=labels2,k=K)
t4<-table(ypca_pred2,Ytest)
classification_report(t4,1)