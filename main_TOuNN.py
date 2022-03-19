import numpy as np
import time
from Mesher import RectangularGridMesher
from material import Material
from TOuNN import TopologyOptimizer
import matplotlib.pyplot as plt
from examples import getExampleBC

ndim = 2; # 2D problem
nelx = 60; # number of FE elements along X
nely = 30; # number of FE elements along Y
elemSize = np.array([1.0,1.0]);
exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely);

mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings);



exampleName = 'TipCantilever'
symMap = {'XAxis':{'isOn':False, 'midPt':10.},\
          'YAxis':{'isOn':False, 'midPt':20.}}


matProp = {'physics':'structural', 'E':1.0, 'nu':0.3}
matProp['penal'] = 1 # SIMP penalization constant, starting value
material = Material(matProp)


# For more BCs see examples.py

nnSettings = {'numLayers':1, 'numNeuronsPerLyr':20 }

fourierMinRadius, fourierMaxRadius = 6, 70;
numTerms = 100;
fourierEncoding = {'isOn':True, 'minRadius':fourierMinRadius, \
              'maxRadius':fourierMaxRadius, 'numTerms':numTerms};
    
    
densityProjection = {'isOn':True, 'sharpness':8};
desiredVolumeFraction = 0.35;

minEpochs = 150; # minimum number of iterations
maxEpochs = 500; # Max number of iterations
    
plt.close('all');
overrideGPU = False
start = time.perf_counter()
topOpt = TopologyOptimizer(exampleName, mesh, material, nnSettings, symMap,\
                            fourierEncoding,desiredVolumeFraction, \
                            densityProjection, overrideGPU);
topOpt.optimizeDesign(maxEpochs,minEpochs);
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
