import numpy as np
import matplotlib.pyplot as plt
from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap
from material import Material
from TOuNN import TOuNN
#%%
ndim, nelx, nely = 2, 40, 20
elemSize = np.array([1., 1.])

exampleName, bcSettings, symMap = getExampleBC(2, nelx, nely);

mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings);


#%% Material
matProp = {'physics':'structural', 'Emax':1.0, 'nu':0.3, 'Emin':1e-3}
material = Material(matProp)

#%% NN
nnSettings = {'outputDim':1, 'numNeuronsPerLayer':20,  'numLayers':2}

fourierMap = {'isOn': True, 'minRadius':4., \
              'maxRadius':80., 'numTerms':  200}

fourierMap['map'] = computeFourierMap(mesh, fourierMap)


#%% Optimization params
lossMethod = {'type':'penalty', 'alpha0':0.05, 'delAlpha':0.05}
#lossMethod = {'type':'logBarrier', 't0':3, 'mu':1.1};

optParams = {'maxEpochs':100, 'learningRate':0.01, 'desiredVolumeFraction':0.5,\
                     'lossMethod':lossMethod}
"""# Run"""
plt.close('all')
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap)
convgHistory = tounn.optimizeDesign(optParams)