#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Author: Sudipan Saha.

"""

import numpy as np

class saturateImage():
##Defines code for image adjusting/pre-processing

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def saturateSomePercentileMultispectral(self,inputMap,percentileToSaturate):
        inputMap=inputMap.astype(float)
        inputMapNormalized=inputMap
        for iter in range(inputMap.shape[2]):
            inputMapBand=inputMap[:,:,iter]
            inputMapNormalizedBand=(inputMapBand-np.amin(inputMapBand))/(np.percentile(inputMapBand,(100-percentileToSaturate))-np.amin(inputMapBand))
            inputMapNormalizedBand[inputMapNormalizedBand>1]=1
            inputMapNormalized[:,:,iter]=inputMapNormalizedBand
        return inputMapNormalized
    
