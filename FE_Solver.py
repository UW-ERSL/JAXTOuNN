import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

class JAXSolver:
  def __init__(self, mesh, material):
    self.mesh = mesh
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
      u_free = jax.scipy.linalg.solve(K[self.mesh.bc['free'],:][:,self.mesh.bc['free']], \
              self.mesh.bc['force'][self.mesh.bc['free']], sym_pos = True, check_finite=False);
      u = jnp.zeros((self.mesh.ndof))
      u = jax.ops.index_add(u, self.mesh.bc['free'], u_free.reshape(-1)) # homog bc wherev fixed
      return u
    #-----------------------#
    @jit
    def computeCompliance(K, u):
      J = jnp.dot(self.mesh.bc['force'].reshape(-1).T, u)
      return J
    #-----------------------#
    K = assembleK(Y)
    u = solve(K)
    J = computeCompliance(K, u)
    return J
