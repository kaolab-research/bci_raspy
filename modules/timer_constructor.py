import time
timer_type = params['timer_type'] # 'pygame' or 'sleep' or 'busy' or 'hybrid'
dt = params['dt'] # microseconds for each tick. Can be decimal.
verbose = params['verbose'] if 'verbose' in params else False # whether or not to print when there's a timing violation
timer_name = params['timer_name'] if 'timer_name' in params else ''

def busy_sleep(dt):
    t_ref = time.time()
    while time.time() - t_ref < dt:
        pass
    return
def hybrid_sleep(dt, time_sleep_buffer=0.001):
    t_ref = time.time()
    if dt > time_sleep_buffer:
        time.sleep(dt - time_sleep_buffer)
    busy_sleep(t_ref + dt - time.time())
    return

rate = 1000000.0/dt # Hz
if timer_type == 'pygame':
    import pygame # necessary?
    clock = pygame.time.Clock()
elif timer_type == 'sleep':
    sleep_fn = time.sleep
elif timer_type == 'busy':
    sleep_fn = busy_sleep
elif timer_type == 'hybrid':
    sleep_fn = hybrid_sleep
t0 = time.time() # in seconds
tickNo = 0
tickCtr = 0
nViolations = 0
t1 = time.time_ns()