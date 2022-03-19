import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyFourierMap, applySymmetry
from jax.experimental import optimizers

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
    convgHistory = {'epoch':[], 'vf':[], 'J':[]}
    xy = applySymmetry(self.xy, self.symMap)
    if(self.fourierMap['isOn']):
      xy = applyFourierMap(xy, self.fourierMap)

    penal = 1
    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
    opt_state = opt_init(self.topNet.wts)
    opt_update = jit(opt_update)
    
    # fwd once to get J0-scaling param
    density0  = self.topNet.forward(get_params(opt_state), xy)
    J0 = self.FE.objectiveHandle(density0.reshape(-1), penal)
  
    def computeLoss(objective, constraints):
      if(optParams['lossMethod']['type'] == 'penalty'):
        alpha = optParams['lossMethod']['alpha0'] + \
                epoch*optParams['lossMethod']['delAlpha'] # penalty method
        loss = objective
        for c in constraints:
          loss += alpha*c**2
      if(optParams['lossMethod']['type'] == 'logBarrier'):
        t = optParams['lossMethod']['t0']* \
                          optParams['lossMethod']['mu']**epoch
        loss = objective
        for c in constraints:
          if(c < (-1/t**2)):
            psi = -jnp.log(-c)/t
          else:
            psi = t*c - jnp.log(1/t**2)/t + 1/t
          loss += psi
      return loss
        
    # closure function
    def closure(nnwts):
      density  = 0.01 + self.topNet.forward(nnwts, xy)
      volCons = (jnp.mean(density)/optParams['desiredVolumeFraction'])- 1.
      J = self.FE.objectiveHandle(density.reshape(-1), penal)
      return computeLoss(J/J0, [volCons])
    
    # optimization loop
    for epoch in range(optParams['maxEpochs']):
      penal = min(8.0, 1. + epoch*0.02)
      opt_state = opt_update(epoch, \
                  optimizers.clip_grads(jax.grad(closure)(get_params(opt_state)), 1.),\
                  opt_state)
  
      if(epoch%10 == 0):
        convgHistory['epoch'].append(epoch)
        density = self.topNet.forward(get_params(opt_state), xy)

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