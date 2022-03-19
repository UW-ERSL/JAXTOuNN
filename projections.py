import torch
import numpy as np


class Projections:

    def __init__(self, symMap, fourierEncoding, densityProj, device):
        self.symMap = symMap
        self.fourierEncoding = fourierEncoding
        self.densityProj = densityProj
        if(self.fourierEncoding['isOn']):
            coordnMap = np.zeros((2, fourierEncoding['numTerms']))
            for i in range(coordnMap.shape[0]):
                for j in range(coordnMap.shape[1]):
                    coordnMap[i, j] = np.random.choice([-1., 1.]) * \
                        np.random.uniform(1./(2*fourierEncoding['maxRadius']),\
                                          1./(2*fourierEncoding['minRadius']))

            self.fourierEncoding['map'] = \
                torch.tensor(coordnMap).float().to(device)
    #-------------------------#

    def applyFourierEncoding(self, x):
        if(self.fourierEncoding['isOn']):
            c = torch.cos(2*np.pi*torch.matmul(x, self.fourierEncoding['map']))
            s = torch.sin(2*np.pi*torch.matmul(x, self.fourierEncoding['map']))
            xv = torch.cat((c, s), axis=1)
            return xv
        return x
    #--------------------------#

    def applyDensityProjection(self, x):
        if(self.densityProj['isOn']):
            b = self.densityProj['sharpness']
            nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
            x = 0.5*nmr/np.tanh(0.5*b)
        return x
    #--------------------------#

    def applySymmetry(self, x):
        if(self.symMap['YAxis']['isOn']):
            xv = self.symMap['YAxis']['midPt'] + \
                torch.abs(x[:, 0] - self.symMap['YAxis']['midPt'])
        else:
            xv = x[:, 0]
        if(self.symMap['XAxis']['isOn']):
            yv = self.symMap['XAxis']['midPt'] + \
                torch.abs(x[:, 1] - self.symMap['XAxis']['midPt'])
        else:
            yv = x[:, 1]
        x = torch.transpose(torch.stack((xv, yv)), 0, 1)
        return x
    #--------------------------#
    