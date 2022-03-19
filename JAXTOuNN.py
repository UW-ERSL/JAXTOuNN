
# We begin by importing the necessary libraries
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, random, jacfwd, value_and_grad
from jax.ops import index, index_add, index_update
from jax.experimental import stax, optimizers
from functools import partial
import jax.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap, applyFourierMap, applySymmetry
from network import TopNet
from FE_Solver import JAXSolver
from material import Material
#-----------------------#

class TOuNN:
  def __init__(self, exampleName, mesh, material, nnSettings, symMap, fourierMap):
    self.exampleName = exampleName
    self.FE = JAXSolver(mesh, material)
    self.xy = self.FE.mesh.elemCenters
    self.fourierMap = fourierMap
    if(fourierMap['isOn']):
      nnSettings['inputDim'] = 2*fourierMap['numTerms']
    else:
      nnSettings['inputDim'] = self.FE.mesh.ndim
    self.topNet = TopNet(nnSettings)
    
    self.symMap = symMap
    #-----------------------#
  
  def optimizeDesign(self, optParams):
    convgHistory = {'vf':[], 'J':[]}
    if(self.fourierMap['isOn']):
      xy = applyFourierMap(self.xy, self.fourierMap)

    penal = 1
    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
    opt_state = opt_init(self.topNet.wts)
    opt_update = jit(opt_update)
    
    # fwd once to get J0-scaling param
    density0  = self.topNet.forward(get_params(opt_state), xy)
    J0 = self.FE.objectiveHandle(density0.reshape(-1), penal)
  
  
    # loss function
    def computeLoss(nnwts):
      penal = min(8.0, 1. + epoch*0.02)
      density  = 0.01 + self.topNet.forward(nnwts, xy)
      # Y = getYoungsModulus(density)
      volcons = (jnp.mean(density)/optParams['desiredVolumeFraction'])- 1.
      J = self.FE.objectiveHandle(density.reshape(-1), penal)
  
      if(optParams['lossMethod']['type'] == 'penalty'):
        alpha = optParams['lossMethod']['alpha0'] + \
                epoch*optParams['lossMethod']['delAlpha'] # penalty method
        loss = J/J0 + alpha*volcons**2;
      if(optParams['lossMethod']['type'] == 'logBarrier'):
        t = optParams['lossMethod']['t0']* \
                          optParams['lossMethod']['mu']**epoch
        if(volcons < (-1/t**2)):
          psi = -jnp.log(-volcons)/t
        else:
          psi = t*volcons - jnp.log(1/t**2)/t + 1/t
        loss = J/J0 + psi
  
      return loss;
    
    # optimization loop
    for epoch in range(optParams['maxEpochs']):
      opt_state = opt_update(epoch, \
                  optimizers.clip_grads(jax.grad(computeLoss)(get_params(opt_state)), 1.),\
                  opt_state)
  
      if(epoch%10 == 0):
        density = self.topNet.forward(get_params(opt_state), xy)
        # Y = getYoungsModulus(density)
        J = self.FE.objectiveHandle(density.reshape(-1), penal)
        convgHistory['J'].append(J)
        volf= jnp.mean(density)
        convgHistory['vf'].append(volf)
        if(epoch == 10):
          J0 = J;
        status = 'epoch {:d}, J {:.2E}, vf {:.2F}'.format(epoch, J/J0, volf);
        print(status)
        if(epoch%30 == 0):
          self.FE.mesh.plotFieldOnMesh(density, status)
    return convgHistory

#%%
ndim, nelx, nely = 2, 40, 20
elemSize = np.array([1., 1.])

exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely);

mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings);


# observe that xyElems is an array from jax. 
# This makes tracking the variable possible
xyElems = jnp.array(mesh.generatePoints())
xyElems = applySymmetry(xyElems, symMap)
print(xyElems.shape)

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