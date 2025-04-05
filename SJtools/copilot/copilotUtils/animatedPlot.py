import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnimatedReward:
    """ shows plt of reward across time"""

    def __init__(self,dim=1,ymin=-10,ymax=[100,1000],labels=None,n=50):
        """
        dim = number of plot in a single graph
        ymin,ymax = self explanatory
        labeles = for creating legends
        n = 10-100. number of data point to show in the graph at any given time
        """
    

        self.dim = dim
        self.time = np.arange(1)
        self.counter = 1
        self.y = [[0] for _ in range(dim)]
        self.line = [None for _ in range(dim)]
        if labels is None: labels = [None for _ in range(dim)]
        self.showN = n
        self.ymin = ymin # graph's minimum scale
        self.ymax = ymax # graph's maximum scale (can have multiple value in case it exceeds the first)
        self.color = ['r-','g-','b-','c-','m-','y-','k-','w-']


        """ comination """
        plt.ion()
        self.fig = plt.figure()
        for i in range(dim): self.line[i], = plt.plot(self.time, self.y[i], self.color[i],label=labels[i]) # Returns a tuple of line objects, thus the comma
        plt.axis([(self.counter-self.showN), 100+(self.counter-self.showN), self.ymin, self.ymax[0]])
        plt.legend()


    def animationUpdate(self):

        """ getting sensor data and printing immediately"""

        # plot data
        for i in range(self.dim): 
            self.line[i].set_data(self.time, self.y[i]) # set data for each line graph
        self.fig.canvas.flush_events() #refresh canvas
        
        # choose ymax to use from self.ymax list using maximum reward we got
        maxv = np.max(self.y)
        ymax = min(self.ymax[-1:]+[v for v in self.ymax if v > maxv])

        # display data
        plt.axis([(self.counter-self.showN), 100+(self.counter-self.showN), self.ymin, ymax])

    
    def valueUpdate(self,r):
    
        self.counter += 1

        """ use this to update any values you have"""
        if self.dim == 1 and not hasattr(r, '__iter__'): r = [r]

        # update time
        if self.counter <= self.showN: self.time = np.arange(self.counter)
        else: self.time += 1

        # update value
        if self.counter <= self.showN: 
            for i in range(self.dim): 
                self.y[i].append(r[i])
        else:
            for i in range(self.dim): 
                self.y[i] = self.y[i][1:]+[r[i]]


if __name__ == "__main__":
    ar = AnimatedReward(dim=2)
    count = 0
    import time
    past = time.time()
    while True:
        # print time it takes for 50 ms
        count += 1
        if count % 50 == 0: 
            print(time.time()-past)
            past = time.time()

        ar.animationUpdate()
        ar.valueUpdate([np.random.random(),np.random.random()])