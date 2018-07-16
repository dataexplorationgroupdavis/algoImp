# python3

# support vector machine from scratch

# KKT conditions:
#   ai = 0 => yi ( wt xi + b) >= 1
#   ai = C => yi ( wt xi + b) <= 1
#   0 < ai < C => yi ( wt xi + b) = 1

import numpy as np 

def predict(Xtest, w, b):
    classification = np.sign(np.dot(np.array(Xtest), w) + b)
    return classification 

def smo(Xtrain, ytrain, C, tol, max_passes):
    n, p = Xtrain.shape

    # Initialization 
    alphas = np.zeros(n)
    b = 0
    passes = 0

    while(passes < max_passes):
        count = 0
        for i in range(n):
            # Ei = f(xi) - yi
            Ei = b - ytrain[i]
            for m in range(n):
                Ei += alphas[m]*ytrain[m]*Xtrain[m].dot(Xtrain[i])
            
            if((ytrain[i]*Ei<-tol and alphas[i]<C) or (ytrain[i]*Ei>tol and alphas[i]>0)):

                # select j != i randomly
                if(i!=0): j=i-1
                else: j=i+1
                
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
               
                if(ita >= 0): continue

                # compute new value for aj
                alphas[j] = aj - ytrain[i]*(Ei - Ej) / ita 
                # clip new value for aj
                if(alphas[j] > H):
                    alphas[j] = H
                
                if(alphas[j] < L):
                    alphas[j] = L
                
                if(abs(aj - alphas[j]) < 10**(-5)): continue

                alphas[i] = ytrain[i]*ytrain[j]*(ai - alphas[i])

                b1 = b - Ei - ytrain[i]*(alphas[i] - ai) * Xtrain[i].dot(Xtrain[i]) - ytrain[i]*(alphas[j] - aj)*Xtrain[i].dot(Xtrain[j]) 
                b2 = b - Ej - ytrain[i]*(alphas[i] - ai) * Xtrain[i].dot(Xtrain[j]) - ytrain[j]*(alphas[j] - aj)*Xtrain[j].dot(Xtrain[j]) 

                if(0 < alphas[i] < C):
                    b = b1
                elif(0 < alphas[j] and alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2

                count += 1

        if( count == 0):
            passes += 1
        else:
            passes = 0

    return alphas, b

def svm(Xtrain, ytrain, C, tol, max_passes, Xtest):
    alphas, b = smo(Xtrain, ytrain, 1, 0.001, 4)
    n, p = Xtrain.shape
    
    w = np.zeros(p)
    for m in range(n):
        w += alphas[m] * ytrain[m] * Xtrain[m]
    
    predictions = predict(Xtest, w, b)
    return predictions 


