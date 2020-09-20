# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import math
import matplotlib.gridspec as gridspec
import pickle as pickle
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer23
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer8
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer10
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer11
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer5
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer2
from skimage.transform import resize
from skimage import filters
from skimage import morphology
import cv2 as cv
from kmodes.kmodes import KModes
import PIL
import cv2
from scipy.spatial.distance import cdist
import scipy.stats as sistats
from saturateSomePercentile import saturateImage
from options import optionsDCVA

##Parsing options
opt = optionsDCVA().parseOptions()
dataPath = opt.dataPath
inputChannels = opt.inputChannels
outputLayerNumbers = np.array(opt.layersToProcess.split(','),dtype=np.int)
thresholdingStrategy = opt.thresholding
otsuScalingFactor = opt.otsuScalingFactor
objectMinSize = opt.objectMinSize
topPercentSaturationOfImageOk=opt.topPercentSaturationOfImageOk
topPercentToSaturate=opt.topPercentToSaturate
multipleCDBool=opt.multipleCDBool
changeVectorBinarizationStrategy=opt.changeVectorBinarizationStrategy
clusteringStrategy=opt.clusteringStrategy
clusterNumber=opt.clusterNumber
hierarchicalDistanceStrategy=opt.hierarchicalDistanceStrategy



nanVar=float('nan')

#Defining parameters related to the CNN
sizeReductionTable=[nanVar,nanVar,1,nanVar,nanVar,2,nanVar,nanVar,4,nanVar,\
                            4,4,4,4,4,4,4,4,4,
                            nanVar,2,nanVar,nanVar,1,nanVar,nanVar,1,1] 
featurePercentileToDiscardTable=[nanVar,nanVar,90,nanVar,nanVar,90,nanVar,nanVar,95,nanVar,\
                            95,95,95,95,95,95,95,95,95,nanVar,95,nanVar,nanVar,95
                            ,nanVar,nanVar,0,0]
filterNumberTable=[nanVar,nanVar,64,nanVar,nanVar,128,nanVar,nanVar,256,nanVar,\
                            256,256,256,256,256,256,256,256,256,nanVar,128,nanVar,nanVar,64,nanVar,nanVar,1,1]

#here "0", the starting index is reflectionPad2D which is not a real layer. So when 
#later operations like filterNumberForOutputLayer=filterNumberTable[outputLayerNumber] are taken, it works, as 0 is dummy and indexing starts from 1

#Reading Image
try:
    inputDataContents=sio.loadmat(dataPath)
    preChangeImage=inputDataContents['preChangeImage']
    postChangeImage=inputDataContents['postChangeImage']
except:
    sys.exit('Cannot read the file. Check if it is a valid .mat file with both pre-change (variable preChangeImage) and post-change data (variable postChangeImage)')





#Pre-change and post-change image normalization
if topPercentSaturationOfImageOk:
    preChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(preChangeImage, topPercentToSaturate)
    postChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(postChangeImage, topPercentToSaturate)
    


#Reassigning pre-change and post-change image to normalized values
data1=np.copy(preChangeImageNormalized)
data2=np.copy(postChangeImageNormalized)

#Checking image dimension
imageSize=data1.shape
imageSizeRow=imageSize[0]
imageSizeCol=imageSize[1]
imageNumberOfChannel=imageSize[2]

if imageSizeRow!=imageSizeCol:
    sys.exit('This code is written for square images. However, code can be easily modified for images with other aspect ratio, however this is\
 not provided here. or the image can be resized to square size')


#Initilizing net / model (G_B: acts as feature extractor here)
input_nc=imageNumberOfChannel #input number of channels
output_nc=6 #from Potsdam dataset number of classes
ngf=64 # number of gen filters in first conv layer
norm_layer = nn.BatchNorm2d
use_dropout=False



netForFeatureExtractionLayer23=ResnetFeatureExtractor9FeatureFromLayer23(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionLayer11=ResnetFeatureExtractor9FeatureFromLayer11(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionLayer10=ResnetFeatureExtractor9FeatureFromLayer10(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionLayer8=ResnetFeatureExtractor9FeatureFromLayer8(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionLayer5=ResnetFeatureExtractor9FeatureFromLayer5(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
netForFeatureExtractionLayer2=ResnetFeatureExtractor9FeatureFromLayer2(input_nc, output_nc, ngf, norm_layer, use_dropout, 9)
if inputChannels=='RGB':
    state_dict=torch.load('./trainedNet/RGB/trainedModelFinal')
    if imageNumberOfChannel!=3:
        sys.exit('Input images do not have 3 channels while loaded model is for R-G-B input')
elif inputChannels=='RGBNIR':
    state_dict=torch.load('./trainedNet/RGBIR/trainedModelFinal')
    if imageNumberOfChannel!=4:
        sys.exit('Input images do not have 4 channels while loaded model is for R-G-B-NIR input')
else:
    sys.exit('Image channels not valid - valid arguments RGB or RGBNIR')
state_dict=torch.load('./trainedNet/RGBIR/trainedModelFinal') 

#for name, param in state_dict.items():
#    print(name)

netForFeatureExtractionLayer23Dict=netForFeatureExtractionLayer23.state_dict()
state_dictForLayer23=state_dict
state_dictForLayer23={k: v for k, v in netForFeatureExtractionLayer23Dict.items() if k in state_dictForLayer23}

netForFeatureExtractionLayer11Dict=netForFeatureExtractionLayer11.state_dict()
state_dictForLayer11=state_dict
state_dictForLayer11={k: v for k, v in netForFeatureExtractionLayer11Dict.items() if k in state_dictForLayer11}

netForFeatureExtractionLayer10Dict=netForFeatureExtractionLayer10.state_dict()
state_dictForLayer10=state_dict
state_dictForLayer10={k: v for k, v in netForFeatureExtractionLayer10Dict.items() if k in state_dictForLayer10}

netForFeatureExtractionLayer8Dict=netForFeatureExtractionLayer8.state_dict()
state_dictForLayer8=state_dict
state_dictForLayer8={k: v for k, v in netForFeatureExtractionLayer8Dict.items() if k in state_dictForLayer8}

netForFeatureExtractionLayer5Dict=netForFeatureExtractionLayer5.state_dict()
state_dictForLayer5=state_dict
state_dictForLayer5={k: v for k, v in netForFeatureExtractionLayer5Dict.items() if k in state_dictForLayer5}

netForFeatureExtractionLayer2Dict=netForFeatureExtractionLayer2.state_dict()
state_dictForLayer2=state_dict
state_dictForLayer2={k: v for k, v in netForFeatureExtractionLayer2Dict.items() if k in state_dictForLayer2}

netForFeatureExtractionLayer23.load_state_dict(state_dictForLayer23)
netForFeatureExtractionLayer11.load_state_dict(state_dictForLayer11)
netForFeatureExtractionLayer10.load_state_dict(state_dictForLayer10)
netForFeatureExtractionLayer8.load_state_dict(state_dictForLayer8)
netForFeatureExtractionLayer5.load_state_dict(state_dictForLayer5)
netForFeatureExtractionLayer2.load_state_dict(state_dictForLayer2)


input_nc=imageNumberOfChannel #input number of channels
output_nc=imageNumberOfChannel #output number of channels
ngf=64 # number of gen filters in first conv layer
norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
use_dropout=False



##changing all nets to eval mode
netForFeatureExtractionLayer23.eval()
netForFeatureExtractionLayer23.requires_grad=False

netForFeatureExtractionLayer11.eval()
netForFeatureExtractionLayer11.requires_grad=False

netForFeatureExtractionLayer10.eval()
netForFeatureExtractionLayer10.requires_grad=False

netForFeatureExtractionLayer8.eval()
netForFeatureExtractionLayer8.requires_grad=False

netForFeatureExtractionLayer5.eval()
netForFeatureExtractionLayer5.requires_grad=False

netForFeatureExtractionLayer2.eval()
netForFeatureExtractionLayer2.requires_grad=False


torch.no_grad()


eachPatch=imageSizeRow
numImageSplitRow=imageSizeRow/eachPatch
numImageSplitCol=imageSizeCol/eachPatch
cutY=list(range(0,imageSizeRow,eachPatch))
cutX=list(range(0,imageSizeCol,eachPatch))
additionalPatchPixel=64


layerWiseFeatureExtractorFunction=[nanVar,nanVar,netForFeatureExtractionLayer2,nanVar,nanVar,netForFeatureExtractionLayer5,nanVar,nanVar,netForFeatureExtractionLayer8,nanVar,\
                            netForFeatureExtractionLayer10,netForFeatureExtractionLayer11,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,nanVar,\
                            nanVar,nanVar,nanVar,netForFeatureExtractionLayer23,nanVar,nanVar,nanVar,nanVar]

 

##Checking validity of feature extraction layers
validFeatureExtractionLayers=[2,5,8,10,11,23] ##Feature extraction from only these layers have been defined here
for outputLayer in outputLayerNumbers:
    if outputLayer not in validFeatureExtractionLayers:
        sys.exit('Feature extraction layer is not valid, valid values are 2,5,8,10,11,23')
        
##Extracting bi-temporal features
modelInputMean=0.406
for outputLayerIter in range(0,len(outputLayerNumbers)):
    outputLayerNumber=outputLayerNumbers[outputLayerIter]
    filterNumberForOutputLayer=filterNumberTable[outputLayerNumber]
    featurePercentileToDiscard=featurePercentileToDiscardTable[outputLayerNumber]
    featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
    sizeReductionForOutputLayer=sizeReductionTable[outputLayerNumber]
    patchOffsetFactor=int(additionalPatchPixel/sizeReductionForOutputLayer)
    print('Processing layer number:'+str(outputLayerNumber))
    
    timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
    timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
    for kY in range(0,len(cutY)):
        for kX in range(0,len(cutX)):
            
            #extracting subset of image 1
            if (kY==0 and kX==0):
                patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                               cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
            elif kY==0:
                patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                               (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
            elif kX==0:
                patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                          (cutY[kY]+eachPatch),\
                                               cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
            else:
                patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                          (cutY[kY]+eachPatch),\
                                          (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
            #extracting subset of image 2   
            if (kY==0 and kX==0):
                patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                               cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
            elif kY==0:
                patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                               (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
            elif kX==0:
                patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                          (cutY[kY]+eachPatch),\
                                               cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
            else:
                patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                          (cutY[kY]+eachPatch),\
                                          (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
           
            #converting to pytorch varibales and changing dimension for input to net
            patchToProcessDate1=patchToProcessDate1-modelInputMean
            
            inputToNetDate1=torch.from_numpy(patchToProcessDate1)
            inputToNetDate1=inputToNetDate1.float()
            inputToNetDate1=np.swapaxes(inputToNetDate1,0,2)
            inputToNetDate1=np.swapaxes(inputToNetDate1,1,2)
            inputToNetDate1=inputToNetDate1.unsqueeze(0)
            
            
            patchToProcessDate2=patchToProcessDate2-modelInputMean
            
            inputToNetDate2=torch.from_numpy(patchToProcessDate2)
            inputToNetDate2=inputToNetDate2.float()
            inputToNetDate2=np.swapaxes(inputToNetDate2,0,2)
            inputToNetDate2=np.swapaxes(inputToNetDate2,1,2)
            inputToNetDate2=inputToNetDate2.unsqueeze(0)
            
            
            #running model on image 1 and converting features to numpy format
               
            with torch.no_grad():
                obtainedFeatureVals1=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
            obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
            obtainedFeatureVals1=obtainedFeatureVals1.data.numpy()
            
            #running model on image 2 and converting features to numpy format
            with torch.no_grad():
                obtainedFeatureVals2=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
            obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
            obtainedFeatureVals2=obtainedFeatureVals2.data.numpy()
            #this features are in format (filterNumber, sizeRow, sizeCol)
            
            
            ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
            obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
            obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)
            
            
            #obtaining features from image 1: resizing and truncating additionalPatchPixel
            if (kY==0 and kX==0):
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals1[processingFeatureIter,\
                                                               0:int(eachPatch/sizeReductionForOutputLayer),\
                                                               0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                               (eachPatch,eachPatch))
                
            elif kY==0:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals1[processingFeatureIter,\
                                                               0:int(eachPatch/sizeReductionForOutputLayer),\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor)],\
                                                               (eachPatch,eachPatch))
            elif kX==0:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals1[processingFeatureIter,\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor),\
                                                               0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                               (eachPatch,eachPatch))
            else:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals1[processingFeatureIter,\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor),\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor)],\
                                                               (eachPatch,eachPatch))
            #obtaining features from image 2: resizing and truncating additionalPatchPixel
            if (kY==0 and kX==0):
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals2[processingFeatureIter,\
                                                               0:int(eachPatch/sizeReductionForOutputLayer),\
                                                               0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                               (eachPatch,eachPatch))
                
            elif kY==0:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals2[processingFeatureIter,\
                                                               0:int(eachPatch/sizeReductionForOutputLayer),\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor)],\
                                                               (eachPatch,eachPatch))
            elif kX==0:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals2[processingFeatureIter,\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor),\
                                                               0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                               (eachPatch,eachPatch))
            else:
                for processingFeatureIter in range(0,filterNumberForOutputLayer):
                    timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                   cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                   resize(obtainedFeatureVals2[processingFeatureIter,\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor),\
                                                               (patchOffsetFactor):\
                                                               (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor)],\
                                                               (eachPatch,eachPatch))
                                   
                                   
    timeVectorDifferenceMatrix=timeVector1Feature-timeVector2Feature
    
    nonZeroVector=[]
    stepSizeForStdCalculation=int(imageSizeRow/2)
    for featureSelectionIter1 in range(0,imageSizeRow,stepSizeForStdCalculation):
        for featureSelectionIter2 in range(0,imageSizeCol,stepSizeForStdCalculation):
            timeVectorDifferenceSelectedRegion=timeVectorDifferenceMatrix\
                                               [featureSelectionIter1:(featureSelectionIter1+stepSizeForStdCalculation),\
                                                featureSelectionIter2:(featureSelectionIter2+stepSizeForStdCalculation),
                                                0:filterNumberForOutputLayer]
            stdVectorDifferenceSelectedRegion=np.std(timeVectorDifferenceSelectedRegion,axis=(0,1))
            featuresOrderedPerStd=np.argsort(-stdVectorDifferenceSelectedRegion)   #negated array to get argsort result in descending order
            nonZeroVectorSelectedRegion=featuresOrderedPerStd[0:featureNumberToRetain]
            nonZeroVector=np.union1d(nonZeroVector,nonZeroVectorSelectedRegion)
            
            
    modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
    modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]
    
    
    ##Normalize the features (separate for both images)
    meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))      
    stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
    normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image
    
    meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))      
    stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
    normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image
    
    ##feature aggregation across channels
    if outputLayerIter==0:
        timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
        timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
    else:
        timeVector1FeatureAggregated=np.concatenate((timeVector1FeatureAggregated,normalizedModifiedTimeVector1),axis=2)
        timeVector2FeatureAggregated=np.concatenate((timeVector2FeatureAggregated,normalizedModifiedTimeVector2),axis=2)
    
 
    
    
del obtainedFeatureVals1, obtainedFeatureVals2, timeVector1Feature, timeVector2Feature, inputToNetDate1, inputToNetDate2 
del netForFeatureExtractionLayer5, netForFeatureExtractionLayer8, netForFeatureExtractionLayer10, netForFeatureExtractionLayer11,netForFeatureExtractionLayer23   
    
absoluteModifiedTimeVectorDifference=np.absolute(saturateImage().saturateSomePercentileMultispectral(timeVector1FeatureAggregated,5)-\
saturateImage().saturateSomePercentileMultispectral(timeVector2FeatureAggregated,5)) 


#take absolute value for binary CD
detectedChangeMap=np.linalg.norm(absoluteModifiedTimeVectorDifference,axis=(2))
detectedChangeMapNormalized=(detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
#plt.figure()
#plt.imshow(detectedChangeMapNormalized)


#detectedChangeMapNormalized=filters.gaussian(detectedChangeMapNormalized,3) #this one is with constant sigma
cdMap=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
if thresholdingStrategy == 'adaptive':
    for sigma in range(101,202,50):
        adaptiveThreshold=2*filters.gaussian(detectedChangeMapNormalized,sigma)
        cdMapTemp=(detectedChangeMapNormalized>adaptiveThreshold) 
        cdMapTemp=morphology.remove_small_objects(cdMapTemp,min_size=objectMinSize)
        cdMap=cdMap | cdMapTemp
elif thresholdingStrategy == 'otsu':
    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
    cdMap = (detectedChangeMapNormalized>otsuThreshold) 
    cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
elif thresholdingStrategy == 'scaledOtsu':
    otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
    cdMap = (detectedChangeMapNormalized>otsuScalingFactor*otsuThreshold) 
    cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
else: 
    sys.exit('Unknown thresholding strategy')
cdMap=morphology.binary_closing(cdMap,morphology.disk(3))

##Creating directory to save result
resultDirectory = './result/'
if not os.path.exists(resultDirectory):
    os.makedirs(resultDirectory)

#Saving the Binary CD result (a .mat file and a .png file)
sio.savemat(resultDirectory+'binaryCdResult.mat', mdict={'cdMap': cdMap})
plt.imsave(resultDirectory+'binaryCdResult.png',np.repeat(np.expand_dims(cdMap,2),3,2).astype(float))






##Multiple CD analysis
if multipleCDBool==True:  ##Multiple CD is performed only if this Bool is True

    #finding indices of changed pixels
    changePixelsAreaAnalyzedIndices=np.where(cdMap)
    changePixelsAreaAnalyzedRow=changePixelsAreaAnalyzedIndices[0]
    changePixelsAreaAnalyzedCol=changePixelsAreaAnalyzedIndices[1]
    changePixelsAreaAnalyzedLinearIndices=np.ravel_multi_index(changePixelsAreaAnalyzedIndices,cdMap.shape)
    numberOfChangePixels=changePixelsAreaAnalyzedRow.shape[0]
    
    #calculating deep change vector and taking out the changed pixels
    modifiedTimeVectorDifference=timeVector1FeatureAggregated-timeVector2FeatureAggregated #absolute difference is not taken here
    modifiedTimeVectorDifferenceSeries=np.reshape(modifiedTimeVectorDifference,\
                                                  (modifiedTimeVectorDifference.shape[0]*modifiedTimeVectorDifference.shape[1],modifiedTimeVectorDifference.shape[2]))
    modifiedTimeVectorDifferenceForChangedPixels=np.take(modifiedTimeVectorDifferenceSeries,changePixelsAreaAnalyzedLinearIndices,axis=0)
    
    
    ##binarizing the deep change vector
    if changeVectorBinarizationStrategy=='zeroThreshold':
        ##simple binarization with 0 as threshold
        modifiedTimeVectorDifferenceForChangedPixelsBinarized=(modifiedTimeVectorDifferenceForChangedPixels>0)
    elif changeVectorBinarizationStrategy=='otsuThreshold':
        ##binarization with Otsu's threshold
        modifiedTimeVectorDifferenceForChangedPixelsBinarized=np.zeros(modifiedTimeVectorDifferenceForChangedPixels.shape,dtype='bool')
        for changeVectorBinarizerIter in range(modifiedTimeVectorDifferenceForChangedPixels.shape[1]):
            modifiedTimeVectorDifferenceForChangedPixelsBinarized[:,changeVectorBinarizerIter]=\
                                                                  modifiedTimeVectorDifferenceForChangedPixels[:,changeVectorBinarizerIter]>\
                                                                  filters.threshold_otsu(modifiedTimeVectorDifferenceForChangedPixels[:,changeVectorBinarizerIter])
    else:
        sys.exit('Change vector binarization strategy not identified. Multiple CD aborted.')
    
    
    
    
    
    if clusteringStrategy == 'kmodes':
        ##applying KModes clustering on the binarized data
        kmodeClusterizer = KModes(n_clusters=clusterNumber, init='Huang')
        resultCluster = kmodeClusterizer.fit_predict(modifiedTimeVectorDifferenceForChangedPixelsBinarized)
        resultCluster=resultCluster+1  #index starts from 0, making it from 1
        
    elif clusteringStrategy == 'hierarchical':
        #applying hierarchical clustering
        
        if hierarchicalDistanceStrategy=='hamming':
        ##hamming distance
            correlationMatrix=1-np.absolute(cdist(np.transpose(modifiedTimeVectorDifferenceForChangedPixelsBinarized),\
                                          np.transpose(modifiedTimeVectorDifferenceForChangedPixelsBinarized),'hamming'))
        elif hierarchicalDistanceStrategy=='correlation':
            ##correlation distance
            ##for correlation distance 0 - perfect correlation, 1 - no correlation, 2 - perfect anticorrelation
            ##need to change to +1, -1 (rather than True, False which is interpreted as 1,0)
            modifiedTimeVectorDifferenceForChangedPixelsBinarizedForCorrDist=\
                                                      np.zeros(modifiedTimeVectorDifferenceForChangedPixelsBinarized.shape,dtype='int')
            modifiedTimeVectorDifferenceForChangedPixelsBinarizedForCorrDist\
                                                      [modifiedTimeVectorDifferenceForChangedPixelsBinarized==True]=1   
            modifiedTimeVectorDifferenceForChangedPixelsBinarizedForCorrDist\
                                                      [modifiedTimeVectorDifferenceForChangedPixelsBinarized==False]=-1                                           
            correlationMatrix=np.absolute(1-cdist(np.transpose(modifiedTimeVectorDifferenceForChangedPixelsBinarizedForCorrDist),\
                                          np.transpose(modifiedTimeVectorDifferenceForChangedPixelsBinarizedForCorrDist),'correlation'))
        else:
            sys.exit('Hierarchical clustering distance measure not recognized. Multiple CD aborted')
            
        #correlationMatrixBinarized=correlationMatrix>(filters.threshold_otsu(correlationMatrix)*2.5)
        correlationMatrixBinarized=correlationMatrix>0.5
        importancePerFeature=np.sum(correlationMatrix,axis=0)  #sum of each column
        tempValImportancePerFeature=importancePerFeature
        featureImportanceOrder=np.zeros([0],dtype='uint8')
        while np.max(tempValImportancePerFeature)!=0:
            mostImpFeature=np.asarray(np.where(tempValImportancePerFeature==np.max(tempValImportancePerFeature)))
            featureImportanceOrder=np.append(featureImportanceOrder, mostImpFeature)
            tempValImportancePerFeature[mostImpFeature]=0
            tempValImportancePerFeature[np.asarray(np.where(correlationMatrixBinarized[mostImpFeature,:]))]=0
          
        outputClusterNumber=clusterNumber
        numberOfFeatureToConsider=int(np.ceil(np.log2(4)))
        featureToConsider=featureImportanceOrder[0:numberOfFeatureToConsider]
        binarizedFeaturesOnlyImpFeatures=modifiedTimeVectorDifferenceForChangedPixelsBinarized[:,featureToConsider]
        binarizedFeaturesOnlyImpFeaturesToLsb=np.zeros([np.shape(binarizedFeaturesOnlyImpFeatures)[0],8],dtype='bool')
        binarizedFeaturesOnlyImpFeaturesToLsb[:,8-numberOfFeatureToConsider:8]=binarizedFeaturesOnlyImpFeatures
        decimalFeature=np.packbits(binarizedFeaturesOnlyImpFeaturesToLsb,axis=1)
          
        
        if (2**numberOfFeatureToConsider-outputClusterNumber)!=0:  ##** acts like ^ 
            numberOfClusterToDiscard=2**numberOfFeatureToConsider-outputClusterNumber
            currentClusterNumber=2**numberOfFeatureToConsider
            for clusterDiscardIter in range(0,numberOfClusterToDiscard):
                clusterModeArray=np.zeros([currentClusterNumber,np.shape(binarizedFeaturesOnlyImpFeatures)[1]])
                clusterSizeArray=np.zeros(currentClusterNumber)
                for currentClusterModeCalculatorIter in range(0,currentClusterNumber):
                    ##took only imp features instead of all features in Matlab version
                    binarizedFeaturesInThisCluster=binarizedFeaturesOnlyImpFeatures\
                                                    [(np.asarray(np.where(decimalFeature==currentClusterModeCalculatorIter)))[0,:],:]
                    clusterModeArray[currentClusterModeCalculatorIter,:]=np.asarray(sistats.mode(binarizedFeaturesInThisCluster,axis=0)[0])
                    clusterSizeArray[currentClusterModeCalculatorIter]=(binarizedFeaturesInThisCluster.shape[0])/numberOfChangePixels
                distanceBetweenClusters=cdist(clusterModeArray,clusterModeArray,'hamming')
                scaledDistanceBetweenClusters=np.copy(distanceBetweenClusters)
                for clusterDistanceScalingIterRow in range (distanceBetweenClusters.shape[0]):
                    for clusterDistanceScalingIterCol in range (distanceBetweenClusters.shape[1]):
                        scaledDistanceBetweenClusters[clusterDistanceScalingIterRow,clusterDistanceScalingIterCol]=\
                                               scaledDistanceBetweenClusters[clusterDistanceScalingIterRow,clusterDistanceScalingIterCol]\
                                               *clusterSizeArray[clusterDistanceScalingIterRow]*clusterSizeArray[clusterDistanceScalingIterCol]
                minDistanceBetweenClusters=np.amin(scaledDistanceBetweenClusters[scaledDistanceBetweenClusters!=0])
                clusterToMerge1=np.where(scaledDistanceBetweenClusters==minDistanceBetweenClusters)[0][0]
                clusterToMerge2=np.where(scaledDistanceBetweenClusters==minDistanceBetweenClusters)[1][0]
                decimalFeature[decimalFeature==clusterToMerge1]=clusterToMerge2
                currentClusterNumber=currentClusterNumber-1
                currentClusters=np.unique(decimalFeature)
                decimalFeatureNew=np.copy(decimalFeature)
                for decimalFeatureValReplaceIter in range(len(currentClusters)):
                    decimalFeatureNew[decimalFeature==currentClusters[decimalFeatureValReplaceIter]]=decimalFeatureValReplaceIter
                decimalFeature=np.copy(decimalFeatureNew)
        
        decimalFeature=decimalFeature+1 #adding 1 so that 0 of changed region does not get mixed with 0 of the background
        resultCluster=np.copy(decimalFeature)
    else:
        sys.exit('Clustering strategy not recognized. Multiple CD aborted')
    
    
    #obtaining multiple CD result image
    multipleChangeOutputMap=np.zeros(cdMap.shape,dtype='uint8')
    for multipleChangeOutputIter in range(0,len(resultCluster)):
        multipleChangeOutputMap[changePixelsAreaAnalyzedRow[multipleChangeOutputIter],\
                                changePixelsAreaAnalyzedCol[multipleChangeOutputIter]]=resultCluster[multipleChangeOutputIter]
            
    #Applying mode filtering on the result
    multipleChangeOutputMap= (filters.rank.modal(multipleChangeOutputMap, morphology.disk(3),mask=cdMap))*cdMap
    
    ##Assigning colors to the multiple CD output map for saving as RGB
    labelColours = np.random.randint(255,size=(100,3))
    labelColours[0,:]=[0,0,0]
    labelColours[1,:]=[255,0,0]
    labelColours[2,:]=[0,0,255]
    labelColours[3,:]=[0,255,0]
    labelColours[4,:]=[255,0,255]
    labelColours[5,:]=[255,255,20]
    labelColours[6,:]=[255,100,20]
    labelColours[7,:]=[100,100,255]
    labelColours[8,:]=[100,255,255]
    multipleChangeOutputMapRGB = np.array([labelColours[ c ] for c in multipleChangeOutputMap])
    
    ##Saving the multiple CD output maps (a .mat file and a .png file)
    sio.savemat(resultDirectory+'./multipleCdResult.mat', mdict={'multipleChangeOutputMap': multipleChangeOutputMap})
    cv.imwrite(resultDirectory+'multipleCdResult.png',multipleChangeOutputMapRGB)


