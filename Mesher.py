import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

class RectangularGridMesher:
  #--------------------------#
  def __init__(self, ndim, nelx, nely, elemSize, bcSettings):
    self.meshType = 'gridMesh'
    self.ndim = ndim;
    self.nelx = nelx;
    self.nely = nely;
    self.elemSize = elemSize;
    self.bcSettings = bcSettings;
    self.numElems = self.nelx*self.nely;
    self.elemArea = self.elemSize[0]*self.elemSize[1]*\
                    jnp.ones((self.numElems)) # all same areas for grid
    self.totalMeshArea = jnp.sum(self.elemArea);
    self.numNodes = (self.nelx+1)*(self.nely+1);
    self.nodesPerElem = 4; # grid quad mesh
    self.ndof = self.bcSettings['dofsPerNode']*self.numNodes;
    self.edofMat, self.nodeIdx, self.elemNodes, self.nodeXY, self.bb = \
                                    self.getMeshStructure();
    self.elemCenters = self.generatePoints();
    self.processBoundaryCondition();
    self.BMatrix = self.getBMatrix(0., 0.)
    self.fig, self.ax = plt.subplots()
  #--------------------------#
  def getBMatrix(self, xi, eta):
    dx, dy = self.elemSize[0], self.elemSize[1];
    B = np.zeros((3,8));
    r1 = np.array([(2.*(eta/4. - 1./4.))/dx, -(2.*(eta/4. - 1./4))/dx,\
                    (2.*(eta/4. + 1./4))/dx,\
                    -(2.*(eta/4. + 1./4))/dx]).reshape(-1);
    r2 = np.array([(2.*(xi/4. - 1./4))/dy, -(2.*(xi/4. + 1./4))/dy,\
                    (2.*(xi/4. + 1./4))/dy, -(2.*(xi/4. - 1./4))/dy])
    
    B = [[r1[0], 0., r1[1], 0., r1[2], 0., r1[3], 0.],\
          [0., r2[0], 0., r2[1], 0., r2[2], 0., r2[3]],\
          [r2[0], r1[0], r2[1], r1[1], r2[2], r1[2], r2[3], r1[3]]];

    return jnp.array(B)
  #--------------------------#
  def getMeshStructure(self):
    # returns edofMat: array of size (numElemsX8) with
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries
    n = self.bcSettings['dofsPerNode']*self.nodesPerElem;
    edofMat=np.zeros((self.nelx*self.nely,n),dtype=int)
    if(self.bcSettings['dofsPerNode'] == 2): # as in structural
      for elx in range(self.nelx):
        for ely in range(self.nely):
          el = ely+elx*self.nely
          n1=(self.nely+1)*elx+ely
          n2=(self.nely+1)*(elx+1)+ely
          edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2,\
                          2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1]);
              
    elif(self.bcSettings['dofsPerNode'] == 1): # as in thermal
      nodenrs = np.reshape(np.arange(0, self.ndof), \
                           (1+self.nelx, 1+self.nely)).T;
      edofVec = np.reshape(nodenrs[0:-1,0:-1]+1, \
                           self.numElems,'F');
      edofMat = np.matlib.repmat(edofVec,4,1).T + \
            np.matlib.repmat(np.array([0, self.nely+1, self.nely, -1]),\
                             self.numElems,1);
    
    iK = tuple(np.kron(edofMat,np.ones((n,1))).flatten().astype(int))
    jK = tuple(np.kron(edofMat,np.ones((1,n))).flatten().astype(int))
    nodeIdx = (iK,jK)


    elemNodes = np.zeros((self.numElems, self.nodesPerElem));
    for elx in range(self.nelx):
      for ely in range(self.nely):
        el = ely+elx*self.nely
        n1=(self.nely+1)*elx+ely
        n2=(self.nely+1)*(elx+1)+ely
        elemNodes[el,:] = np.array([n1+1, n2+1, n2, n1])
    bb = {}
    bb['xmin'],bb['xmax'],bb['ymin'],bb['ymax'] = \
        0., self.nelx*self.elemSize[0],\
        0., self.nely*self.elemSize[1]
        
    nodeXY = np.zeros((self.numNodes, 2))
    ctr = 0;
    for i in range(self.nelx+1):
      for j in range(self.nely+1):
        nodeXY[ctr,0] = self.elemSize[0]*i;
        nodeXY[ctr,1] = self.elemSize[1]*j;
        ctr += 1;
            
    return edofMat, nodeIdx, elemNodes, nodeXY, bb
  #--------------------------#

  def generatePoints(self, res=1):
    # args: Mesh is dictionary containing nelx, nely, elemSize...
    # res is the number of points per elem
    # returns an array of size (numpts X 2)
    xy = np.zeros((res**2*self.numElems, 2))
    ctr = 0
    for i in range(res*self.nelx):
      for j in range(res*self.nely):
        xy[ctr, 0] = (i + 0.5)/(res*self.elemSize[0])
        xy[ctr, 1] = (j + 0.5)/(res*self.elemSize[1])
        ctr += 1
    return xy
  #--------------------------#
  def processBoundaryCondition(self):
    force = np.zeros((self.ndof,1))
    dofs=np.arange(self.ndof)
    fixed = dofs[self.bcSettings['fixedNodes']]
    free = np.setdiff1d(np.arange(self.ndof), fixed)
    force[self.bcSettings['forceNodes']] = self.bcSettings['forceMagnitude']
    self.bc = {'force':force, 'fixed':fixed,'free':free}
  #--------------------------#
  def plotFieldOnMesh(self, field, titleStr):
    plt.ion(); plt.clf()
    plt.imshow(-np.flipud(field.reshape((self.nelx,self.nely)).T), \
               cmap='gray', interpolation='none')
    plt.axis('Equal')
    plt.grid(False)
    plt.title(titleStr)
    plt.pause(0.01)
    self.fig.canvas.draw()
