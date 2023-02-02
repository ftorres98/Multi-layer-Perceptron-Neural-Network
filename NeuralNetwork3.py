'''
Fernando Torres
'''

import numpy as np
import sys
import math


class MLP:
    def __init__(self, batchSize, learningRate, regularRate, epochNum, trainingData, trainingLabel):
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.regularRate = regularRate
        self.epochNum = epochNum
        self.trainingData = trainingData
        self.trainingLabel = trainingLabel

        self.InputLayerSize = 4
        self.OutputLayerSize = 2
        self.HiddenLayerSize1 = 6

        self.W1 = np.random.randn(self.HiddenLayerSize1,self.InputLayerSize)*np.sqrt(1.0/self.InputLayerSize)
        self.B1 = np.random.randn(self.HiddenLayerSize1,1)*np.sqrt(1.0/self.InputLayerSize)
        self.W2 = np.random.randn(self.OutputLayerSize,self.HiddenLayerSize1)*np.sqrt(1.0/self.HiddenLayerSize1)
        self.B2 = np.random.randn(self.OutputLayerSize,1)*np.sqrt(1.0/self.HiddenLayerSize1)

    def crossEntropy(self, y, prob):
        num = y.shape[1]

        cost = (1.0/num)*np.sum(np.multiply(y, np.log(prob)) + np.multiply(1-y,np.log(1-prob)))
        return -cost

    def accuracy(self, x, y):
        predictions = []

        results = self.forwardPass(x)
        pred = np.argmax(results['sig2'], axis=0)
        predictions.append(pred == np.argmax(y,axis=0))

        print(np.mean(predictions))


    def softMax(self, x):
        num = np.exp(x - x.max())
        sum = np.sum(num,axis=0)
        return num / sum 

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return z*(1-z)

    def updateWeights(self, feedBackward):
        self.W1 = self.W1 - (self.learningRate * feedBackward['deltaW1'])
        self.B1 = self.B1 - (self.learningRate*feedBackward['deltaB1'])
        self.W2 = self.W2 - (self.learningRate * feedBackward['deltaW2'])
        self.B2 = self.B2 - (self.learningRate*feedBackward['deltaB2'])

    def miniBatch(self, x, y):
        num = x.shape[1]
        miniBatches = list()
        numOfBatches = math.floor(num/self.batchSize)

        for i in range(numOfBatches):
            deltaX = x[:, i*self.batchSize : (i+1)*self.batchSize]
            deltaY = y[:, i*self.batchSize : (i+1)*self.batchSize]

            miniBatch = (deltaX, deltaY)
            miniBatches.append(miniBatch)

        if num % self.batchSize != 0:
            deltaX = x[:, self.batchSize*numOfBatches : num]
            deltaY = y[:, self.batchSize*numOfBatches : num]

            miniBatch = (deltaX, deltaY)
            miniBatches.append(miniBatch)

        return miniBatches

    def forwardPass(self, x):
        feedForward = dict()

        #1st hidden layer
        feedForward['forward1'] = np.dot(self.W1, x) + self.B1
        feedForward['sig1'] = self.sigmoid(feedForward['forward1'])

        #output layer
        feedForward['forward2'] = np.dot(self.W2,feedForward['sig1']) + self.B2
        feedForward['sig2'] = self.softMax(feedForward['forward2'])

        return feedForward
        
    def backwardProp(self, x, y, feedForward):
        feedBackward = dict()


        batchSize = x.shape[1]

        delta = feedForward['sig2'] - y

        feedBackward['deltaW2'] = (1.0/batchSize) * np.dot(delta, feedForward['sig1'].T)
        feedBackward['deltaB2'] = (1.0/batchSize) * np.sum(delta,axis=1,keepdims=True)

        feedBackward['deltaSig1'] = np.dot(self.W2.T, delta)
        feedBackward['deltaForward1'] = feedBackward['deltaSig1'] * self.sigmoidPrime(feedForward['sig1'])

        feedBackward['deltaW1'] = (1.0/batchSize) * np.dot(feedBackward['deltaForward1'], x.T)
        feedBackward['deltaB1'] = (1.0/batchSize) * np.sum(feedBackward['deltaForward1'],axis=1,keepdims=True)

        feedBackward['deltaW1'] = feedBackward['deltaW1'] + (self.regularRate*self.W1)
        feedBackward['deltaW2'] = feedBackward['deltaW2'] + (self.regularRate*self.W2)

        return feedBackward
    
    def train(self):
        for i in range(self.epochNum):
            ran = np.arange(self.trainingData.shape[1])
            np.random.shuffle(ran)

            deltaX = self.trainingData[:,ran]
            deltaY = self.trainingLabel[:,ran]

            miniBatches = self.miniBatch(deltaX, deltaY)

            for miniBatch in miniBatches:
                x, y = miniBatch
                forwardResults = self.forwardPass(x)
                backwardResults = self.backwardProp(x, y, forwardResults)
                self.updateWeights(backwardResults)

            #self.accuracy(self.trainingData, self.trainingLabel)


if __name__ == '__main__':
    def readIN():
        trainData = []
        trainLabel = []
        testData = []

        trainData = np.loadtxt(sys.argv[1], dtype=float, delimiter =',')
        trainLabel = np.loadtxt(sys.argv[2], dtype=int)
        testData = np.loadtxt(sys.argv[3], dtype=float, delimiter=',')

        return np.array(trainData, ndmin=2), np.array(trainLabel, ndmin=2), np.array(testData,ndmin=2)

    def writeOUT(output, path='test_predictions.csv'):
        with open(path, 'w') as f:
            for i in output:
                f.write(str(i) + '\n')
            

    trainingData, trainingLabel, testingData = readIN()

    encodedArray = np.zeros((trainingLabel.size, trainingLabel.max()+1), dtype=int)
    encodedArray[np.arange(trainingLabel.size),trainingLabel] = 1

    mlp = MLP(32,0.1,0.001,2000,np.concatenate((trainingData, trainingData**2), axis=1).T, encodedArray.T)

    mlp.train()

    results = mlp.forwardPass(np.concatenate((testingData, testingData**2), axis=1).T)

    pred = np.argmax(results['sig2'], axis=0)

    writeOUT(pred)

