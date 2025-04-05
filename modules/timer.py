if timer_type == 'pygame':
    clock.tick_busy_loop(rate)
else:
    tickCtr += 1
    t_i = t0 + (dt*tickCtr)/1000000.0 # expected time of next tick in seconds
    t_diff = t_i - time.time() # time from now until next tick.
    if t_diff > 0:
        sleep_fn(t_diff)
    else: # start new alignment if timing not met
        t0 = time.time()
        tickCtr = 0
        if tickNo > 1: # expect a violation on 1st tick
            nViolations += 1
            if verbose:
                print('Warning: timing violation on tick #', tickNo, 'by', -t_diff, 'seconds')
tickNo += 1
try:
    # This is useful for when there is only one timer,
    # but has limited use when there are multiple timers.
    # Needs to be int64 or larger.
    timer_tick_time_ns[:] = time.time_ns()
except:
    pass