import jax.numpy as jnp
import numpy as np


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
    c = jnp.cos(2*np.pi*jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
    s = jnp.sin(2*np.pi*jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
    xy = jnp.concatenate((c, s), axis = 1)
  return xy

#-------DENSITY PROJECTION-----------#

def applyDensityProjection(x, densityProj):
  if(densityProj['isOn']):
    b = densityProj['sharpness']
    nmr = np.tanh(0.5*b) + jnp.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
  return x

#-------SYMMETRY-----------#
def applySymmetry(x, symMap):
  if(symMap['YAxis']['isOn']):
    xv = x[:,0].at[:].set(symMap['YAxis']['midPt']\
                          + jnp.abs( x[:,0] - symMap['YAxis']['midPt']) )

  else:
    xv = x[:, 0]
  if(symMap['XAxis']['isOn']):
    yv = x[:,1].at[:].set(symMap['XAxis']['midPt']\
                          + jnp.abs( x[:,1] - symMap['XAxis']['midPt']) )
  else:
    yv = x[:, 1]
  x = jnp.stack((xv, yv)).T
  return x
#--------------------------#