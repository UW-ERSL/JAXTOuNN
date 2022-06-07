import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

class JAXSolver:
  def __init__(self, mesh, material):
    self.mesh = mesh
    self.material = material
    self.objectiveHandle = jit(self.objective)
    self.D0 = self.material.getD0elemMatrix(self.mesh)
  #-----------------------# 
  def objective(self, density, penal):
    @jit
    def getYoungsModulus(density, penal):
      Y = self.material.matProp['Emin'] + \
            (self.material.matProp['Emax']-self.material.matProp['Emin'])*\
              (density+0.001)**penal
      return Y
    #-----------------------#
    @jit
    def assembleK(Y):
      K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
      sK = jnp.einsum('e, jk->ejk', Y, self.D0).flatten()
      K = K.at[self.mesh.nodeIdx].add(sK)
      return K
    #-----------------------#
    @jit
    def solve(K):
      # eliminate fixed dofs for solving sys of eqns
      u_free = jax.scipy.linalg.solve(K[self.mesh.bc['free'],:][:,self.mesh.bc['free']], \
              self.mesh.bc['force'][self.mesh.bc['free']], sym_pos = True, check_finite=False);
      u = jnp.zeros((self.mesh.ndof))
      u = u.at[self.mesh.bc['free']].add(u_free.reshape(-1)) # homog bc wherev fixed
      return u
    #-----------------------#
    @jit
    def computeCompliance(K, u):
      J = jnp.dot(self.mesh.bc['force'].reshape(-1).T, u)
      return J
    #-----------------------#
    Y = getYoungsModulus(density, penal)
    K = assembleK(Y)
    u = solve(K)
    J = computeCompliance(K, u)
    return J
