
retval = game.poll()
if not retval:
    quit_ = True
game.set_vel(kf_sent_vel)
game.draw()
pygame.display.flip()