
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

rand_key = random.PRNGKey(0) # reproducibility


ndim, nelx, nely = 2, 40, 20
elemSize = np.array([1., 1.])

exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely);

mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings);


# observe that xyElems is an array from jax. 
# This makes tracking the variable possible
xyElems = jnp.array(mesh.generatePoints())
print(xyElems.shape)

"""### Material"""

#Next we define the relevant material property. 
# We are concerned only with structural mech
# at the moment. penal here refers to the SIMP penalization constant
material = {'Emax':1., 'Emin':1e-3, 'nu':0.3, 'penal':1.}

# with the material defined, we can now calculate the base constitutive matrix
def getD0(material):
  # the base constitutive matrix assumes unit 
  #area element with E = 1. and nu prescribed.
  # the material is also assumed to be isotropic.
  # returns a matrix of size (8X8)
  E = 1.
  nu = material['nu'];
  k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,\
                  -1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
  KE = \
  E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
  [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
  [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
  [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
  [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
  [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
  [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
  [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
  return KE

material['D0'] = getD0(material)


"""### Symmetry

The resulting structure might be symmetric about an axis. However, owing to the nonlinearity of the NN this may not be enforced implicitly. We therefore explicitly enforce symmetry by transforming the coordinates

"""

"""### Neural Network"""

# Let us now define the actual NN. We consider a fully connected NN
# with LeakyRelu as the activation and a sigmoid in the output layer
def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun
Swish = elementwise(nn.swish)

def makeNetwork(nnspec):
  # JAX network definition
  layers = []
  for i in range(nnspec['numLayers']-1):
    layers.append(stax.Dense(nnspec['numNeuronsPerLayer']))
    layers.append(Swish)#(stax.LeakyRelu)
  layers.append(stax.Dense(nnspec['outputDim']))
  layers.append(stax.Sigmoid)
  return stax.serial(*layers)

nnspec = {'outputDim':1, 'numNeuronsPerLayer':20,  'numLayers':2}

"""### FE Solver CM

We now turn our attention to defining functions that are needed for solving the system. We use jit to speed up the computation
"""

class FESolver:
  def __init__(self, mesh, material, bc):
    self.mesh = mesh
    self.bc = bc
    self.material = material
    self.objectiveHandle = jit(self.objective)
  #-----------------------# 
  def objective(self, Y):
    @jit
    def assembleK(Y):
      K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
      kflat_t = (self.material['D0'].flatten()[np.newaxis]).T 
      sK = (kflat_t*Y).T.flatten()
      K = jax.ops.index_add(K, self.mesh.nodeIdx, sK)
      return K
    #-----------------------#
    @jit
    def solve(K):
      # eliminate fixed dofs for solving sys of eqns
      u_free = jax.scipy.linalg.solve(K[self.bc['free'],:][:,self.bc['free']], \
              self.bc['force'][self.bc['free']], sym_pos = True, check_finite=False);
      u = jnp.zeros((self.mesh.ndof))
      u = jax.ops.index_add(u, self.bc['free'], u_free.reshape(-1)) # homog bc wherev fixed
      return u
    #-----------------------#
    @jit
    def computeCompliance(K, u):
      J = jnp.dot(self.bc['force'].reshape(-1).T, u)
      return J
    #-----------------------#
    K = assembleK(Y)
    u = solve(K)
    J = computeCompliance(K, u)
    return J

"""# Opt

### Projections

Input and output projections help us define among many geometric, manufacturing constraints.
"""

#-------FOURIER LENGTH SCALE-----------#
def computeFourierMap(mesh, fourierMap):
  # compute the map
  coordnMapSize = (mesh.ndim, fourierMap['numTerms']);
  freqSign = np.random.choice([-1.,1.], coordnMapSize)
  stdUniform = np.random.uniform(0.,1., coordnMapSize) 
  wmin = 1./(2*fourierMap['maxRadius']*mesh.elemSize[0])
  wmax = 1./(2*fourierMap['minRadius']*mesh.elemSize[0]) # w~1/R
  wu = wmin +  (wmax - wmin)*stdUniform
  coordnMap = np.einsum('ij,ij->ij', freqSign, wu)
  return coordnMap
#-----------------#
def applyFourierMap(xy, fourierMap):
  if(fourierMap['isOn']):
    c = jnp.cos(2*np.pi*jnp.einsum('ij,jk->ik', xyElems, fourierMap['map']))
    s = jnp.sin(2*np.pi*jnp.einsum('ij,jk->ik', xyElems, fourierMap['map']))
    xy = jnp.concatenate((c, s), axis = 1)
  return xy

"""### Optimization
Finally, we are now ready to express the optimization problem
"""

# Optimization params
lossMethod = {'type':'penalty', 'alpha0':0.05, 'delAlpha':0.05}
#lossMethod = {'type':'logBarrier', 't0':3, 'mu':1.1};

fourierMap = {'isOn': True, 'minRadius':4., \
              'maxRadius':80., 'numTerms':  200}

fourierMap['map'] = computeFourierMap(mesh, fourierMap)


optimizationParams = {'maxEpochs':450, 'learningRate':0.01, 'desiredVolumeFraction':0.5,\
                     'lossMethod':lossMethod}

def optimizeDesign(xy, optParams, mesh, material, bc, fourierMap):
  FE = FESolver(mesh, material, bc)
  # input projection
  if(fourierMap['isOn']):
   xy = applyFourierMap(xy, fourierMap)
  # make the NN
  init_fn, applyNN = makeNetwork(nnspec);
  fwdNN = jit(lambda nnwts, x: applyNN(nnwts, x))
  _, params = init_fn(rand_key, (-1, xy.shape[1]))
  # optimizer
  opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
  opt_state = opt_init(params)
  opt_update = jit(opt_update)
  
  # fwd once to get J0-scaling param
  density0  = fwdNN(get_params(opt_state), xy)
  J0 = FE.objectiveHandle(density0.reshape(-1))

  def getYoungsModulus(density):
    material['penal'] = min(8., 1. + epoch*0.02)
    Y = material['Emin'] + \
          (material['Emax']-material['Emin'])*(density+0.001)**material['penal']
    return Y
  #-----------------------#
  # loss function
  def computeLoss(nnwts):
    density  = 0.01 + fwdNN(nnwts, xy)
    Y = getYoungsModulus(density)
    volcons = (jnp.mean(density)/optParams['desiredVolumeFraction'])- 1.
    J = FE.objectiveHandle(Y.reshape(-1))

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
      density = fwdNN(get_params(opt_state), xy)
      Y = getYoungsModulus(density)
      J = FE.objectiveHandle(Y.reshape(-1))
      volf= jnp.mean(density)
      if(epoch == 10):
        J0 = J;
      status = 'epoch {:d}, J {:.2E}, vf {:.2F}'.format(epoch, J/J0, volf);
      print(status)
      if(epoch%30 == 0):
        plt.figure();
        plt.imshow(-jnp.flipud(density.reshape((nelx, nely)).T),\
                  cmap='gray')
        plt.title(status)
        plt.pause(0.001)
        plt.show()

  return fwdNN, get_params(opt_state)

"""# Run"""
bc = mesh.bc
network, nnwts = optimizeDesign(xyElems, optimizationParams, mesh, material, bc, fourierMap)