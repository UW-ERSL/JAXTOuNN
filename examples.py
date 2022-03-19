import numpy as np
import time
import matplotlib.pyplot as plt

    #  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
def getExampleBC(example, nelx, nely):
    if(example == 1): # tip cantilever
        exampleName = 'TipCantilever'
        bcSettings = {'fixedNodes': np.arange(0,2*(nely+1),1),\
                      'forceMagnitude': -1.,\
                      'forceNodes': 2*(nelx+1)*(nely+1)-2*nely+1, \
                      'dofsPerNode':2};
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx}}
    
    elif(example == 2): # mid cantilever
        exampleName = 'MidCantilever'
        bcSettings = {'fixedNodes': np.arange(0,2*(nely+1),1),\
                      'forceMagnitude': -1.,\
                      'forceNodes': 2*(nelx+1)*(nely+1)- (nely+1),\
                      'dofsPerNode':2};
        symMap = {'XAxis':{'isOn':True, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx}}
    
    elif(example == 3): #  MBBBeam
        exampleName = 'MBBBeam'
        fn= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
        bcSettings = {'fixedNodes': fn,\
                      'forceMagnitude': -1.,\
                      'forceNodes': 2*(nely+1)+1,\
                      'dofsPerNode':2};  
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx}}
    
    elif(example == 4): #  Michell
        exampleName = 'Michell'
        fn = np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely] )
        bcSettings = {'fixedNodes': fn,\
                      'forceMagnitude': -1.,\
                      'forceNodes': nelx*(nely+1)+2,\
                      'dofsPerNode':2};  
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':True, 'midPt':0.5*nelx}}
    
    elif(example == 5): #  DistributedMBB
        exampleName = 'Bridge'
        fixn = np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
        frcn = np.arange(2*nely+1, 2*(nelx+1)*(nely+1), 8*(nely+1))
        bcSettings = {'fixedNodes': fixn,\
                      'forceMagnitude': -1./(nelx+1.),\
                      'forceNodes': frcn ,\
                      'dofsPerNode':2};  
        symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely},\
          'YAxis':{'isOn':True, 'midPt':0.5*nelx}}
    elif(example == 6): # Tensile bar
        exampleName = 'TensileBar'
        fn =np.union1d(np.arange(0,2*(nely+1),2), 1); 
        midDofX= 2*(nelx+1)*(nely+1)- (nely);
        bcSettings = {'fixedNodes': fn,\
                      'forceMagnitude': 1.,\
                      'forceNodes': midDofX,\
                      'dofsPerNode':2}; 
        symMap = {'XAxis':{'isOn':True, 'midPt':0.5*nely},\
          'YAxis':{'isOn':False, 'midPt':0.5*nelx}}
    
    return exampleName, bcSettings, symMap

    
