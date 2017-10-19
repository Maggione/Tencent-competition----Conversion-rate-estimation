#!/usr/bin/python
# coding=utf-8

import numpy
import random
import scipy.special as special
import math
from math import log
import numpy as np

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
        # print (self.alpha, self.beta)

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
    def __compute_moment(self, tries, success):
        '''moment estimation'''
        click = np.asarray(success)
        pv = np.asarray(tries)
        ctr_list = click / pv
        var = ctr_list.var()
        mean = ctr_list.mean()
        """
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)
        return mean, var/(len(ctr_list)-1)
 """
        return mean, var

    def calibration(slef, tries, success):
        eps = 0.00000001
        return (success + eps + self.alpha)/(tries + self.alpha + self.beta + eps)

def test():
    hyper = HyperParam(1, 1)
    #--------sample training data--------
    I, C = hyper.sample_from_beta(10, 1000, 100, 1000)
    print I,C
    
    #--------estimate parameter using moment estimation--------
    hyper.update_from_data_by_moment(I, C)
    print hyper.alpha, hyper.beta

    #--------estimate parameter using fixed-point iteration--------
    hyper.update_from_data_by_FPI(I, C, 100, 0.00001)
    print hyper.alpha, hyper.beta  
