import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(num=[1, 3, 5, 7, 9]):
    ## valid_digits is a vector containing the digits
    ## we wish to classify.
    ## Do not change anything inside of this function
    data = ds.load_digits()
    labels = data['target']
    features = data['data']
    X = features
    Y = labels
    X = X / np.repeat(np.max(X, axis=1), 64).reshape(X.shape[0], -1)
    for i in range(len(Y)):
      Y[i] = 1 if Y[i] in num else 0

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=10)
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_test = Y_test.reshape((len(Y_test), 1))

    return X_train, Y_train, X_test, Y_test


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train an SVM to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_SVM(X_train, Y_train, X_test, Y_test, kernel=None, lamb=0.01, num_iterations=200, learning_rate=0.1):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## lamb: Regularization parameter
    ## num_iterations: Number of iterations.
    ## learning_rate: Learning rate.

    n,p = X_train.shape
    ntest = X_test.shape[0]
    beta = np.random.randn(p+1,1)
    
    X1 = np.c_[np.repeat(1,n).reshape(n,1),X_train] #shape:n*(p+1)
    X1test = np.c_[np.repeat(1,ntest).reshape(ntest,1),X_test]
    Ytrain = 2 * Y_train-1 #shape:n*1
    Ytest = 2 * Y_test-1
    
    acc_train = np.zeros((num_iterations,1))
    acc_test = np.zeros((num_iterations,1))
    
    if kernel == None:
        for it in range(num_iterations):
            score = np.dot(X1,beta)
            delta = score * Ytrain <1
            dbeta = np.sum(np.repeat(delta * Ytrain,p+1,axis=1)*X1,axis=0).reshape(p+1,1)/n
            beta = beta + learning_rate*dbeta
            beta[1:(p+1)] = beta[1:(p+1)] - lamb*beta[1:(p+1)]
            acc_train[it] = np.mean(np.sign(np.dot(X1, beta))==Ytrain)
            acc_test[it] = np.mean(np.sign(np.dot(X1test,beta))==Ytest)
        #             if it % 10 == 0:
        #                 print("Training Accuracy: ", acc_train[it])
        #                 print("Testing Accuracy: ", acc_test[it])
        Y_train_pre = (np.sign(np.dot(X1, beta))+1)/2
        Y_test_pre = (np.sign(np.dot(X1test, beta))+1)/2
    
    if kernel == 'Gaussian':
        #initialize
        alpha = np.zeros((n,1)).reshape(n,1)
        delta = np.zeros((n,1))
        C=1
        
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = np.exp(-10*np.sum((X1[i,]-X1[j,])**2))
        Ktest = np.zeros((n,ntest))
        for i in range(n):
            for j in range(ntest):
                Ktest[i,j] = np.exp(-10*np.sum((X1[i,]-X1test[j,])**2))
        z = np.dot(K,alpha).reshape(n,1) #shape: n*1
        Q = np.repeat(Ytrain, n, axis = 1)* K* np.repeat(Ytrain.T, n, axis = 0)
        
        # coordinate descent
        for it in range(num_iterations):
            for i in range(n):
                delta[i] = max(-alpha[i], min(C-alpha[i],(1-z[i])/Q[i,i]))
            index = np.argmax(delta)
            #             index = np.random.randint(n)
            alpha[index] = alpha[index] + delta[index]
            z = z + (Q[:,index]*delta[index]).reshape(n,1)
            Y_train_pre = np.sign(np.sum(alpha * Ytrain * K, axis = 0)).reshape(n,1)
            Y_test_pre = np.sign(np.sum(alpha * Ytrain * Ktest, axis = 0)).reshape(ntest,1)
            acc_train[it] = np.mean(Y_train_pre == Ytrain)
            acc_test[it] = np.mean(Y_test_pre == Ytest)
        #             if it % 10 == 0:
        #                 print("Training Accuracy: ", acc_train[it])
        #                 print("Testing Accuracy: ", acc_test[it])
        
        Y_train_pre = (Y_train_pre +1)/2
        Y_test_pre = (Y_test_pre +1)/2


    ## Function should output 3 things:
    ## 1. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 2. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    ## 3. The final predicted Y_train
    ## 4. The final predicted Y_test

    return acc_train, acc_test, Y_train_pre, Y_test_pre

def testing_example():

    X_train, Y_train, X_test, Y_test = prepare_data()

    acc_train, acc_test, Y_trainp, Y_testp = my_SVM(
        X_train, Y_train, X_test, Y_test, num_iterations=1000)

    ax = plt.plot(range(len(acc_test)), acc_train, range(len(acc_test)), acc_test)
    plt.title("Simple SVM")
    plt.xlabel('Number of iteration')
    plt.ylabel('Accuracy')
    plt.legend(('Training Accuracy', 'Testing Accuracy'))
    plt.show()


    acc_train, acc_test, Y_trainp, Y_testp = my_SVM(
        X_train, Y_train, X_test, Y_test, kernel='Gaussian', num_iterations=1000)

    ax = plt.plot(range(len(acc_test)), acc_train, range(len(acc_test)), acc_test)
    plt.title("SVM with Gaussian Kernal")
    plt.xlabel('Number of iteration')
    plt.ylabel('Accuracy')
    plt.legend(('Training Accuracy', 'Testing Accuracy'))
    plt.show()
