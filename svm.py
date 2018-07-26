# python3

# support vector machine from scratch
# using smo algorithm to update alphas

# KKT conditions:
#   ai = 0 => yi ( wt xi + b) >= 1
#   ai = C => yi ( wt xi + b) <= 1
#   0 < ai < C => yi ( wt xi + b) = 1

import numpy as np 
import pdb
from random import randint
import time 

def predict(Xtest, w, b):
    classification = np.sign(np.dot(np.array(Xtest), w) + b)
    return classification 

def smo(Xtrain, ytrain, C, tol, max_passes):
    n, p = Xtrain.shape
    #  print(('n:{},len:{}').format(n, len(ytrain)))

    # Initialization 
    alphas = np.zeros(n)
    b = 0
    passes = 0
    print('smo starts!')

    while(passes < max_passes):
        print(("passes:{}").format(passes))
        passtime = time.time()
        count = 0
        #  pdb.set_trace()
        for i in range(n):
            # Ei is our loss
            Ei = b - ytrain[i]
            for m in range(n):
                Ei += alphas[m]*ytrain[m]*Xtrain[m].dot(Xtrain[i])
            
            if((ytrain[i]*Ei<-tol and alphas[i]<C) or (ytrain[i]*Ei>tol and alphas[i]>0)):

                # select j != i randomly
                j = randint(0,n-1)
                while(j == i): j = randint(0,n-1)

                #  print(("({},{})").format(i,j))
                Ej = b - ytrain[j]
                for m in range(n):
                    Ej += alphas[m]*ytrain[m]*Xtrain[m].dot(Xtrain[j])

                ai = alphas[i]
                aj = alphas[j]

                # compute L and H
                if(ytrain[i] != ytrain[j]):
                    L = max(0, aj-ai)
                    H = min(C, C+aj-ai)
                else:
                    L = max(0, ai+aj-C)
                    H = min(C, ai+aj)

                ita = 2*Xtrain[i].dot(Xtrain[j]) - Xtrain[i].dot(Xtrain[i]) - Xtrain[j].dot(Xtrain[j]) 
                # continue for jump to next i 
                if(ita >= 0): continue

                # compute new value for aj
                alphas[j] = aj - ytrain[j]*(Ei - Ej) / ita 
                
                # clip new value for aj
                if(alphas[j] > H):
                    alphas[j] = H
                
                if(alphas[j] < L):
                    alphas[j] = L
                
                if(abs(aj - alphas[j]) < 10**(-5)): continue

                # update ai
                alphas[i] += ytrain[i]*ytrain[j]*(ai - alphas[i])

                b1 = b - Ei - ytrain[i]*(alphas[i] - ai) * Xtrain[i].dot(Xtrain[i]) - ytrain[i]*(alphas[j] - aj)*Xtrain[i].dot(Xtrain[j]) 
                b2 = b - Ej - ytrain[i]*(alphas[i] - ai) * Xtrain[i].dot(Xtrain[j]) - ytrain[j]*(alphas[j] - aj)*Xtrain[j].dot(Xtrain[j]) 

                if(0 < alphas[i] < C):
                    b = b1
                elif(0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2

                count += 1

        print('count',count)
        print(('pass time: {}s').format(int(time.time() - passtime)))
        if( count == 0):
            passes += 1
        else:
            passes = 0

    return alphas, b

def svm(Xtrain, ytrain, C, tol, max_passes):
    alphas, b = smo(Xtrain, ytrain, C, tol, max_passes)
    print(alphas)
    n, p = Xtrain.shape
    
    w = np.zeros(p)
    for m in range(n):
        w += alphas[m] * ytrain[m] * Xtrain[m]
    
    return w, b

def main():
    from mnist import MNIST                                                                                                             
    mndata = MNIST("../MNIST/samples")                                                                                                         
    Xtrain, ytrain = mndata.load_training()                                                                                                    
    Xtest, ytest = mndata.load_testing() 
 
#   Training Data    
#   make a subset of binary data
    subsetZip = np.array([[j,i] for [j,i] in enumerate(ytrain) if i == 0 or i == 1])
    indice = subsetZip[:,0]
    ytrainNew = subsetZip[:,1]
    ytrainBin = [1 if i == 1 else -1 for i in ytrainNew]
    XtrainNew = np.array(Xtrain)[indice]
#   normalize data 
    Xtrain = np.array(XtrainNew) / 255.0
    ytrain = ytrainBin 
    
#   Testing Data
    subsetZipTest = np.array([[j,i] for [j,i] in enumerate(ytest) if i == 0 or i == 1])
    indiceTest = subsetZipTest[:,0]
    ytestNew = subsetZipTest[:,1]
    ytestBin = [1 if i == 1 else -1 for i in ytestNew]
    #  pdb.set_trace()
    XtestNew = np.array(Xtest)[indiceTest]
#   normalize data 
    Xtest = np.array(XtestNew) / 255.0
    ytest = ytestBin
    Xtest = np.array(Xtest) / 255.0    

    w, b = svm(Xtrain, ytrain, C=5, tol=0.001, max_passes=3)
    predictions = predict(Xtest, w, b)
    print(("model accuracy: {}").format(sum(ytest == predictions) / float(len(ytest))))


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("time spent:", time.time() - starttime)
