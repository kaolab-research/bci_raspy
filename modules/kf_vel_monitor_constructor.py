import pygame
import numpy as np

class Game():
    def __init__(self):
        self.screenSize = np.array([700, 700])
        self.screen = pygame.display.set_mode(self.screenSize)
        self.v = np.zeros(2)
        return
    def set_vel(self, v):
        self.vel = v.copy()
        return
    def poll(self):
        retval = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                retval = False

        return retval
    def draw(self):
        self.screen.fill((0, 0, 0))
        center = 0.5*self.screenSize
        L = 200
        dp = (self.vel*L)*([1, -1])
        pygame.draw.lines(self.screen, (128, 128, 128), True, [center + [-L, -L], center + [L, -L], center + [L, L], center + [-L, L]])
        pygame.draw.line(self.screen, (255, 255, 255), center, center+dp)
        return
    
game = Game()
