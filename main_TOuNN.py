import numpy as np
import matplotlib.pyplot as plt
from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence


import configparser

#%% read config file
configFile = './config.txt'
config = configparser.ConfigParser()
config.read(configFile)

#%% Mesh and BC
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim') # default for 2
nelx = meshConfig.getint('nelx') # number of FE elements along X
nely = meshConfig.getint('nely') # number of FE elements along Y
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)
exampleName, bcSettings, symMap = getExampleBC(2, nelx, nely)
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings)

#%% Material
materialConfig = config['MATERIAL']
E, nu =  materialConfig.getfloat('E'), materialConfig.getfloat('nu')
matProp = {'physics':'structural', 'Emax':E, 'nu':nu, 'Emin':1e-3*E}
material = Material(matProp)

#%% NN
tounnConfig = config['TOUNN']
nnSettings = {'numLayers': tounnConfig.getint('numLayers'),\
              'numNeuronsPerLayer':tounnConfig.getint('hiddenDim'),\
              'outputDim':tounnConfig.getint('outputDim')}
  
fourierMap = {'isOn':tounnConfig.getboolean('fourier_isOn'),\
              'minRadius':tounnConfig.getfloat('fourier_minRadius'), \
              'maxRadius':tounnConfig.getfloat('fourier_maxRadius'),\
              'numTerms':tounnConfig.getint('fourier_numTerms')}

fourierMap['map'] = computeFourierMap(mesh, fourierMap)

#%% Optimization params
lossConfig = config['LOSS']
lossMethod = {'type':'logBarrier', 't0':lossConfig.getfloat('t0'),\
              'mu':lossConfig.getfloat('mu')}
# lossMethod = {'type':'penalty', 'alpha0':lossConfig.getfloat('alpha0'), \
          #     'delAlpha':lossConfig.getfloat('delAlpha')}
          
optConfig = config['OPTIMIZATION']
optParams = {'maxEpochs':optConfig.getint('numEpochs'),\
             'lossMethod':lossMethod,\
             'learningRate':optConfig.getfloat('lr'),\
             'desiredVolumeFraction':optConfig.getfloat('desiredVolumeFraction'),\
             'gradclip':{'isOn':optConfig.getboolean('gradClip_isOn'),\
                         'thresh':optConfig.getfloat('gradClip_clipNorm')}}


#%% Run optimization
plt.close('all')
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap)
convgHistory = tounn.optimizeDesign(optParams)
plotConvergence(convgHistory)