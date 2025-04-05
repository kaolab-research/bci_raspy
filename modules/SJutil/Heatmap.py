import numpy as np
from scipy import ndimage
import pygame

class Heatmap():
    def __init__(self, n, option = []):
        """ 
            option can be ["com","bcc","only","df0.9"]
            "only": indicates that copilot won't observe entire heatmap and will only read com or bcc
            "bcc": bin count at cursor
            "com": center of mass
            "df0.9": decay factor of 0.9
            self.map # should contain n x n heat map 
            self.bcc # getter should get bcc of heatmap
            self.com # getter should get com of heatmap
        """

        self.n = n
        self.dim = self.n * self.n
        self.shape = (self.n,self.n)
        self.map = np.zeros(self.shape)

        self.tickLength = 0.002

        # extracting optional parameters regarding heatmap 
        self.useBcc = "bcc" in option # bin count at cursor
        self.useCom = "com" in option # center of mass
        self.useOnly = "only" in option
        self.decayFactor = 1
        for item in option:
            if item[:2] == "df":
                self.decayFactor = float(item[2:])
                break

        # to be read by things outside
        self.com = np.zeros(2) if self.useCom else None
        self.bcc = 0

        # calculate dimension (integer) to be added to copilot's observation space
        self.copilot_dim = self.calculateCopilotDim(self.useCom, self.useBcc, self.useOnly)

    def calculateCopilotDim(self, useCom, useBcc, useOnly):
        """ used one time to calculate dimension that is to be added by copilot """
        dim = 0 if useOnly else self.n * self.n
        if useCom: dim += 2
        if useBcc: dim += 1
        return dim

    def update_window_size_constants(self, L):
        self.heatmapImgPos = np.zeros(self.shape+(2,))
        lenx = self.shape[0]
        leny = self.shape[1]
        xinc = L / lenx
        yinc = L / leny
        for i in range(lenx):
            for j in range(leny):
                self.heatmapImgPos[i][j] = (xinc * i, yinc * (leny-1-j)) # reversing j because up should be positive and down negative
        self.heatmapImgSize = np.array((xinc,yinc))
        self.heatmapColor = np.array((100,255,100)) * 1

    def reset(self):
        """ resets obs_heatmap to all zeros"""
        self.map = np.zeros(self.shape)

    def update(self, cursorPos):
        """ add one to obs_heatmap at cursorPos """
        n = self.n

        index = (cursorPos + np.array([1,1])) * n // 2
        index = np.clip(index,None,n-1)
        index = (int(index[0]),int(index[1]))
        self.map[index] += self.tickLength # assuming trial length is 500 tick
    
        # exponential decay
        self.map *= self.decayFactor # decay factor

        # bin count
        self.bcc = self.map[index]

        # calculate com
        if self.useCom:
            # note 0,0 index is at bottom left of the map
            ij = ndimage.center_of_mass(self.map)

            # this is how you convert it to xy in [-1,1]x[-1,1]
            com = (2*np.array(ij)+1) / n - 1 
            self.com = com
            

    def draw_state(self,screen):
        # only to be called by draw_state
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                heatmapBlockColor = self.heatmapColor * self.map[i][j]
                heatmapBlockColor = np.clip(heatmapBlockColor,0,255)
                pygame.draw.rect(screen, heatmapBlockColor ,(self.heatmapImgPos[i][j],self.heatmapImgSize),0)
       

if __name__ == "__main__":
    heatmap = Heatmap(n=10,option=["com"])
    print("shape",heatmap.shape)
    heatmap.update(np.array([0.85,0.85]))
    heatmap.update(np.array([0,0.85]))
    print("map",heatmap.map)
    print("com",heatmap.com)