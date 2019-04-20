import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x,w,b):
    return 1.0/(1+np.exp(-(np.dot(x,w)+b)))

def loss():
    pass

class SigmoidNeuron(object):
    def __init__(self):
        self.w = None
        self.b = None
    
    def grad_w(self,x,y):
        return np.dot(np.transpose(x), 2* ( sigmoid(x, self.w, self.b) -y )* sigmoid(x, self.w, self.b)* (1- sigmoid(x, self.w, self.b)) )

    def grad_b(self,x,y):
        return 2* (sigmoid(x, self.w, self.b) -y )* sigmoid(x, self.w, self.b)* (1- sigmoid(x, self.w, self.b)) * 1
        
    def lossf(self,x,y):
        return - (y *(np.log(sigmoid(x,self.w,self.b))) + (1-y)* (np.log(1- sigmoid(x,self.w,self.b))) )
        
    def fit(self,X,Y, initialise = True, epochs =200, eta = 0.1):
        if initialise:
            self.w = np.random.random(X.shape[1])*10 -5
            self.b =np.random.random()*10 -5
            
        print(self.w)
        print(self.b)
            
        self.loss = []
        self.weights = []
        self.B = []
        
        for t in range(epochs):
            #self.loss.append( -np.sum(Y * np.log( sigmoid(X,self.w, self.b)) + (1-Y)* np.log( 1- sigmoid(X,self.w, self.b)) ))
            #print("\nEpoch {0}".format(t))
            #print( (Y - sigmoid(X,self.w, self.b))* (Y - sigmoid(X,self.w, self.b)))
            self.loss.append( np.sum( (Y - sigmoid(X,self.w, self.b))* (Y - sigmoid(X,self.w, self.b))))
            self.weights.append(self.w)
            self.B.append(self.b)
            self.w = self.w - eta * (np.sum(self.grad_w(X,Y)))
            self.b = self.b - eta * (np.sum(self.grad_b(X,Y))) 
            
        plt.plot(self.loss,"*")
        
    def predict(self,X):
        Y = sigmoid(X, self.w, self.b)
        return Y >= 0.5
  
  
  Y = np.array([0,0,0,1,1,1,1])
X = np.array( [ [1,6],[2,4],[3,7],[4,2],[5,7],[6,2],[7,2]])
s = SigmoidNeuron()
s.fit(X,Y)

s.fit(X,Y,initialise = True, eta = 0.01,epochs = 20000)
sigmoid(X,s.w,s.b)
print(s.w, s.b)
