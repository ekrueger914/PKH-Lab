# import libraries for the notebook

import sys
import numpy as np
import tifffile as tif
import matplotlib.pyplot
import ipywidgets
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import rcParams
import skimage
import scipy

#Set global path for files
sys.path.append('/home/ekrueger2/repos/celldetection')

#Import other relevant functions
import transients
import util
from cellpose import models 
import skimage.io
from skimage import exposure

model = models.Cellpose(gpu=False, model_type='cyto') #channels['channel3'].raw=channels['channel3'].raw+1e-4


#Class for container
class container():
  def __init__(self,fileName,
                    index=None,# none- single channel image; int otherwise
                    raw = None # array of data; usually leave undefined 
                    ):
    self.fileName=fileName
    self.index = index
    self.raw = raw
            
# Class for cell properties
class cellProp():
    def __init__(self,coords=None,area=None):
        self.coords=coords
        self.area =area


#Create function to prepare data for cell detection
def preparation(path, fileNames, channel_names):
    channels = {} #create a dictionary to store channel info
    for i,name in enumerate(channel_names):
        channel = container( fileNames[i], index=i)
        channels[name] = channel
        channels[name].channel_index = i
    for channel in channels.values():  # get raw data from images
      ar = transients.LoadTimeData(path+"/"+channel.fileName,
                                 timeReversed=True
                                )
      channel.raw = np.asarray(ar)
    return(channels)

#Create function to perform cell segmentation
def cellsegmentation(imgs,display=False):
    # cell segmentation
    data=imgs[:,:,0]
    # adjust contrast
    dataAdjusted=exposure.adjust_gamma(data, gamma=2.0)

    # plot results
    masks, flows, styles, diams = model.eval(dataAdjusted, diameter=None,do_3D=False)
    np.set_printoptions(threshold=sys.maxsize) # do we need this 

    
    
    #with open ('check', 'w+') as f:
    #    f.write(str(masks))
    if display:
        plt.figure(figsize=(20,10))
        ax1=plt.subplot(131)
        ax1.imshow(data)
        ax2=plt.subplot(132)
        ax2.imshow(dataAdjusted)

        ax3=plt.subplot(133)
        ax3.imshow(masks)

        # label individual cells
        labels=[]
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                if masks[i][j]==0:
                    continue
                else:
                    if masks[i][j] not in labels:
                        labels.append(masks[i][j])
                        ax3.text(j, i, int(masks[i][j])-1,ha="center", va="center", fontsize=10, fontweight='black',color='orange')
                        
                        
        return masks                


#Create function to analyze the masks from the cell segmentation
def celldetectionanalysis(channels, masks):
    # get area for each cell
    numOfCells=np.max(masks)
    areas=[]
    for i in range(1,numOfCells+1):
        count=0
        #print("PKH fix later ")
        #for j in range(masks.shape[0]):
        #    for k in range(masks.shape[1]):
        #        if masks[j][k]==i:
        #            count+=1
        #if iter == 0:
        #    continue
        output = np.where(masks==i) 
#        print(np.shape(output))
        area = np.shape(output)[1]
#        print(area)        
        areas.append(area)

    # get coordinates for each cell
    numOfCells=np.max(masks)

    indices=[]
    for i in range(1,numOfCells+1):
        index=np.argwhere(masks==i)
        indices.append(index)   

 
    indices=np.array(indices)

    # convert masks to binary format
    newMasks=np.zeros_like(masks,int)
    newMasks[masks>=1]=1   

    for key,channel in channels.items():
        result = channel.raw*newMasks[None,:,:]    
        channel.masked = result
        channel.stacked=newMasks             

    # store segmentation info as region_cells using cellProp object    
    region_cells=[]
    for i in range(len(areas)):
        region_cells.append(cellProp(indices[i],areas[i]))

    channels['channel1'].region_cells=region_cells


#Create function to create transients from cell segmentation
def calciumtransients(channels,fileName=None,display=False):
    # get Ca transient traces

    #traces, region_cells = transients.GetTraces(ar,img)

    channel = channels['channel1']
    region_cells = channel.region_cells
    traces, region_cells_MASTER = transients.GetTraces(
                channel.masked, # time-series data after masking 
                channel.stacked,# mask 
                region_cells=channel.region_cells,
                channelName=channel.index)
    channel.traces = traces
    channel.region_cells_MASTER = region_cells_MASTER

    if display:    
        normalize=True #split here for next experiment
        per=6
        nTrace=len(channel.traces)
        nPanels=int(nTrace/per)+1
        count=0

        rcParams['figure.figsize']=20,10
        fig=plt.figure()
        print(nTrace)
        for i in range(nTrace):
            if i % per == 0:
                ax=fig.add_subplot(4,int(nPanels/4)+1,count+1)
                count+=1

            trace=np.copy(channel.traces[i])

            if normalize:
                trace-=np.min(trace)
                trace/=np.max(trace)

            ax.plot(trace,label=i)
            plt.legend(bbox_to_anchor=(1,1))

        if fileName is not None:    
          plt.gcf().savefig(fileName + ".png")


#Create function to screen cells for transient formation
def screencells(channels,minCellSize=300, maxCellSize=None, minFluctuation=100,display=False):
    # Screen out cells that do not meet the criteria (semi-subjective process)
    channel = channels['channel1']
    channel.minCellSize = minCellSize
    channel.maxCellSize = maxCellSize
    channel.minFluctuation = minFluctuation

    newTracesMap = transients.ScreenTraces(
        channel.traces,
        channel.region_cells_MASTER,       # changed from "channel.region_cells" to "region_cells_MASTER"                         
        minCellSize = channel.minCellSize,
        maxCellSize = channel.maxCellSize,
        minFluctuation = channel.minFluctuation, # base on intensity 
        #minDerivative = channel.minDerivative # based on derivative (looking for rapid upswing, not slow increase)                 
    )

    normalize=True
    per=6
    nTrace=len(channel.traces)
    nPanels=int(nTrace/per)+1
    count=0

    if display:
        rcParams['figure.figsize']=20,10
        fig=plt.figure()

        for i,idx in enumerate(newTracesMap):

            if i % per == 0:
                ax=fig.add_subplot(4,int(nPanels/4)+1,count+1)
                count+=1

            trace=np.copy(channel.traces[idx])

            if normalize:
                trace-=np.min(trace)
                trace/=np.max(trace)

            ax.plot(trace,label=idx)
            plt.legend(bbox_to_anchor=(1,1))
    return(newTracesMap)
