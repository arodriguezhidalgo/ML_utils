#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:36:18 2020

@author: arodriguez
"""

class FeatureExtraction:
    def __init__(self, seed = 1492):
        self.seed = seed;
        self.tf = {};
        
    def polynomial(self, degree = 2):
        from sklearn.preprocessing import PolynomialFeatures
        self.tf['poly'] = PolynomialFeatures(degree=degree)     
    
    def fit(self, df):
        '''
        It fits all the available features that are stored in the object.
        '''
        for i in self.tf.keys():
            self.tf[i].fit(df);
    
    def transform(self, df, feature_name):
        '''
        Transformation takes place independently and by-demand.
        '''
        return self.tf[feature_name].transform(df);
    
    def get_transform_list(self):
        '''
        This function shows all the feature extractors that are stored in the object.
        '''
        return self.tf.keys()
    