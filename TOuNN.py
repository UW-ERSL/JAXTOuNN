#Versions
#Numpy 1.18.1
#Pytorch 1.5.0
#scipy 1.4.1
#cvxopt 1.2.0

#%% imports
import numpy as np
import torch
import torch.optim as optim
from FE_Solver import TorchSolver
from projections import Projections
from plotUtil import Plotter
import matplotlib.pyplot as plt
from network import TopNet
from torch.autograd import grad

#%% main TO functionalities
class TopologyOptimizer:
    #-----------------------------#

    def __init__(self, exampleName, mesh, material, nnSettings, symMap,\
                 fourierEncoding, desiredVolumeFraction, densityProj, \
                     overrideGPU = True):

        self.exampleName = exampleName
        self.device = self.setDevice(overrideGPU)
        self.boundaryResolution  = 3 # default value for plotting and interpreting

        self.FE = TorchSolver(mesh, material)
        self.xy = torch.tensor(self.FE.mesh.elemCenters, requires_grad = True).\
                                        float().view(-1,2).to(self.device)
        self.xyPlot = torch.tensor(self.FE.mesh.generatePoints(self.boundaryResolution),\
                        requires_grad = True).float().view(-1,2).to(self.device)
        self.Pltr = Plotter()

        self.desiredVolumeFraction = desiredVolumeFraction
        self.density = self.desiredVolumeFraction*np.ones((self.FE.mesh.numElems))
        self.Projctn = Projections\
            (symMap, fourierEncoding, densityProj, self.device)

        if(fourierEncoding['isOn']):
            inputDim = 2*self.Projctn.fourierEncoding['numTerms'];
        else:
            inputDim = self.FE.mesh.ndim;
        self.topNet = TopNet(nnSettings, inputDim).to(self.device)
        self.objective = 0.
    #-----------------------------#
    def setDevice(self, overrideGPU):
        if(torch.cuda.is_available() and (overrideGPU == False) ):
            device = torch.device("cuda:0")
            print("GPU enabled")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        return device;

    #-----------------------------#
    def optimizeDesign(self,maxEpochs, minEpochs):
        self.convergenceHistory = {'compliance':[], 'vol':[], 'grayElems':[]};
        learningRate = 0.01;
        alphaMax = 100*self.desiredVolumeFraction;
        alphaIncrement= 0.05;
        alpha = alphaIncrement; # start
        nrmThreshold = 0.01; # for gradient clipping
        self.obj0 = 1.;
        self.optimizer = optim.Adam(self.topNet.parameters(), amsgrad=True,lr=learningRate);
        batch_x = self.Projctn.applySymmetry(self.xy);
        x = self.Projctn.applyFourierEncoding(batch_x);
        for epoch in range(maxEpochs):

            self.optimizer.zero_grad();
            nn_rho = torch.flatten(self.topNet(x)).to(self.device);
            nn_rho = self.Projctn.applyDensityProjection(nn_rho);

            uv_displacement = self.FE.solveFE(nn_rho)
            objective = self.FE.computeCompliance(uv_displacement)/self.obj0
            volConstraint =(torch.einsum('i,i->i',nn_rho, self.FE.mesh.elemArea).sum()\
                             /(self.FE.mesh.totalMeshArea*self.desiredVolumeFraction)) - 1.0; # global vol constraint
            currentVolumeFraction = torch.mean(nn_rho).item();
            self.objective = objective;
            if(epoch == 0):
                self.obj0 = objective.item();
            loss =   self.objective+ alpha*torch.pow(volConstraint,2);

            alpha = min(alphaMax, alpha + alphaIncrement);
            loss.backward(retain_graph=True);
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(),nrmThreshold)
            self.optimizer.step();
            if(epoch%20 == 0):
                greyElements= sum(1 for rho in nn_rho if ((rho > 0.2) & (rho < 0.8)));
                relGreyElements = self.desiredVolumeFraction*greyElements/len(nn_rho);
            self.convergenceHistory['compliance'].append(self.objective.item());
            self.convergenceHistory['vol'].append(currentVolumeFraction);
            self.convergenceHistory['grayElems'].append(relGreyElements);
            self.FE.material.matProp['penal'] = min(8.0,self.FE.material.matProp['penal'] + 0.02); # continuation scheme
            titleStr = "Iter {:d} , Obj {:.2F} , vol {:.2F}".format(epoch, self.objective.item()*self.obj0, currentVolumeFraction);
            print(titleStr);
            if(epoch % 20 == 0):
                self.FE.mesh.plotFieldOnMesh(nn_rho.detach().cpu(), titleStr);

            if ((epoch > minEpochs ) & (relGreyElements < 0.025)):
                break;
        self.FE.mesh.plotFieldOnMesh(nn_rho.detach().cpu(), titleStr);

        print("{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} "\
             .format(epoch, self.objective.item()*self.obj0 ,currentVolumeFraction,loss.item(),relGreyElements));

        print("Final J : {:.3f}".format(self.objective.item()*self.obj0));
        self.Pltr.plotConvergence(self.convergenceHistory);

        batch_x = self.Projctn.applySymmetry(self.xyPlot);
        x = self.Projctn.applyFourierEncoding(batch_x);
        rho = torch.flatten(self.Projctn.applyDensityProjection(self.topNet(x)));
        rho_np = rho.cpu().detach().numpy();

        titleStr = "Iter {:d} , Obj {:.2F} , vol {:.2F}".format(epoch, self.objective.item()*self.obj0, currentVolumeFraction);
        self.Pltr.plotDensity(self.xyPlot.detach().cpu().numpy(),\
                  rho_np.reshape((self.FE.mesh.nelx*self.boundaryResolution,\
                  self.FE.mesh.nely*self.boundaryResolution)), titleStr);
        return self.convergenceHistory;