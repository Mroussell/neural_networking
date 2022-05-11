"""
MJR
Michael Roussell
Copyright 2022

A Simple Binary Decision Perceptron Class.
Takes pair of binary inputs and produces prediction based on a nueral network.

Python 3.9.12 version of the python interpreter.
If there are any questions, please contact me at 'mjr.dev.contact@gmail.com.

MIT Education License Preferred.
"""

import numpy
import random
import os

class Perceptron:

    def __init__(self, learning_rate:int=1, bias:int=1, weights:list=None):
        """
        Initialize Perceptron object for binary decision models.

        Parameters
        ----------
        learning_rate : int, default=1
            Determines on what speed the neural network will learn.
        bais : int, default=1
            Value may be added to the total value calculated. 
        weights : list, default=None
            Used to calculate error

        Returns
        -------
        None
        """
        self.learning_rate = learning_rate 
        self.bias = bias
        self.pred = None
        if weights == None: 
            #weights generated in a list (3 weights in total for 2 neurons and the bias)
            self.weights = [random.random(),random.random(),random.random()]
        else:
            self.weights = weights

    
    def work(self, input1, input2, output):
        """
        Calculate work on nueron.

        Parameters
        ----------
        input1 : int
            Value of first neuron.
        input2 : int 
            Value of second neuron.
        output : list, default=None
           Expected output value.

        Returns
        -------
        None
        """
        outputP = (input1 * self.weights[0]) + (input2 * self.weights[1]) + (self.bias * self.weights[2])
        if outputP > 0 : #activation function (here Heaviside)
            outputP = 1
        else :
            outputP = 0
        error = output - outputP
        self.weights[0] += error * input1 * self.learning_rate
        self.weights[1] += error * input2 * self.learning_rate
        self.weights[2] += error * self.bias * self.learning_rate
        

    def fit_or(self, x:int, y:int, scale=50):
        """
        Calculate fit for OR gate.

        Parameters
        ----------
        x : int
            First input.
        y : int
            Second input.
        scale : int
            Range for fitting.

        Returns
        -------
        None
        """
        for i in range(scale):
            self.work(1,1,1) #True or true
            self.work(1,0,1) #True or false
            self.work(0,1,1) #False or true
            self.work(0,0,0) #False or false
        out_p = x*self.weights[0] + y*self.weights[1] + self.bias*self.weights[2]
        self.pred = 1/(1+numpy.exp(-out_p)) #sigmoid function
        self.pred = round(self.pred)

    
    def fit_and(self, x:int, y:int, scale=50):
        """
        Calculate fit for AND gate.

        Parameters
        ----------
        x : int
            First input.
        y : int
            Second input.
        scale : int
            Range for fitting.

        Returns
        -------
        None
        """
        for i in range(scale):
            self.work(1,1,1) #True or true
            self.work(1,0,0) #True or false
            self.work(0,1,0) #False or true
            self.work(0,0,0) #False or false
        out_p = x*self.weights[0] + y*self.weights[1] + self.bias*self.weights[2]
        self.pred = 1/(1+numpy.exp(-out_p)) #sigmoid function
        self.pred = round(self.pred)
        