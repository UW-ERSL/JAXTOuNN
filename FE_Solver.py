#--------------------------#
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.matlib

import torch
from torch_sparse_solve import solve

#--------------------------#
class TorchSolver:

    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material

        self.D0elem = torch.tensor(self.material.getD0elemMatrix(self.mesh))

        self.f = torch.tensor(self.mesh.bc['force']).unsqueeze(0)

        V = np.zeros((mesh.ndof, mesh.ndof));
        V[self.mesh.bc['fixed'],self.mesh.bc['fixed']] = 1.
        V = torch.tensor(V[np.newaxis])
        indices = torch.nonzero(V).t()
        values = V[indices[0], indices[1], indices[2]] # modify this based on dimensionality
        penal = 100000000.*self.material.matProp['E'];
        self.fixedBCPenaltyMatrix = \
            penal*torch.sparse_coo_tensor(indices, values, V.size())

    #--------------------------#

    def assembleK(self, rho):
        E = self.material.computeRAMP_Interpolation(rho);
        sK = torch.einsum('i,ijk->ijk', E, self.D0elem).flatten()
        Kasm = torch.sparse_coo_tensor(self.mesh.nodeIdx, sK, \
                (1, self.mesh.ndof, self.mesh.ndof))
        return Kasm;
    #--------------------------#

    def solveFE(self, rho):
        Kasm = self.assembleK(rho);
        K = (Kasm + self.fixedBCPenaltyMatrix).coalesce()
        u = solve(K, self.f).flatten()
        return u;
    #--------------------------#

    def computeCompliance(self, u):
        J = torch.einsum('i,i->i', u, self.f.view(-1)).sum()
        return J;
    #--------------------------#
