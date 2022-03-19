import numpy as np
import torch

class Material:
    def __init__(self, matProp):
        self.matProp = matProp
        E, nu = matProp['E'], matProp['nu'];
        self.C = E/(1-nu**2)* \
                np.array([[1, nu, 0],\
                          [nu, 1, 0],\
                          [0, 0, (1-nu)/2]]);
    #--------------------------#
    
    def computeSIMP_Interpolation(self, rho):
        E = 0.001*self.matProp['E'] + \
                (0.999*self.matProp['E'])*\
                (rho+0.01)**self.matProp['penal']
        return E
    #--------------------------#
    
    def computeRAMP_Interpolation(self, rho):
        E = 0.001*self.matProp['E']  +\
            (0.999*self.matProp['E'])*\
                (rho/(1.+self.matProp['penal']*(1.-rho)))
        return E
    #--------------------------#
    def getD0elemMatrix(self, mesh):
        if(mesh.meshType == 'gridMesh'):
            E = 1
            nu = self.matProp['nu'];
            k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
                          -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
            D0 = E/(1-nu**2)*np.array\
         ([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
            # all the elems have same base stiffness
            D0elem = np.repeat(D0[np.newaxis, :, :], mesh.numElems, axis=0)
            return D0elem
        elif(mesh.meshType == 'gMesh'):
            D0elem = np.zeros((mesh.numElems, mesh.dofsPerElem, mesh.dofsPerElem))
            for elem in range (mesh.numElems):
              KElemtemp = np.zeros((mesh.dofsPerElem,mesh.dofsPerElem))
              nodes = mesh.elemNodes[elem]
              xNodes = mesh.nodeXY[nodes,0]
              yNodes = mesh.nodeXY[nodes,1]
              invJ = np.array([[-yNodes[0]+yNodes[2], -yNodes[1]+yNodes[0]],\
                          [-xNodes[2]+xNodes[0], -xNodes[0]+xNodes[1]]]);
              dJ = np.linalg.det(invJ)
              invJ = invJ/dJ;
              Z = np.zeros((mesh.nodesPerElem));
              for g in range(len(mesh.gaussWt)):
                N = mesh.shapefn_N[g,:,:]
                gradN = mesh.gradshapefn_B[g,:,:]
                x = xNodes @ N .astype(float);
                T1 = invJ[0,:] @ gradN
                T2 = invJ[1,:] @ gradN
          #     # plane stress follows
                B = np.array([np.concatenate((T1, Z)), np.concatenate((Z, T2)), \
                              np.concatenate((T2, T1))]);
                KElemtemp = KElemtemp + \
                            mesh.gaussWt[g] * dJ * np.transpose(B) @ self.C @ B;

              order = [0, 3, 1, 4, 2, 5];
              KElemtemp = KElemtemp[:,order]
              KElemtemp = KElemtemp[order,:]
              D0elem[elem,:,:] = KElemtemp
            return D0elem
        #--------------------------#
        
        #--------------------------#
            
        
        
        
        
        
        
        
        
    
    