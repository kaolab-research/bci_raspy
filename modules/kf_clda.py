
rluds_output[0:len(RLUDs_idx)] = decoder_output[[RLUDs_idx]]
# convert from [right, left, up, down] to signed [xvel, yvel] 
signed_vel = np.array([rluds_output[0] - rluds_output[1], rluds_output[2] - rluds_output[3]])

kf_state_cache = kf_state.copy()

kf_state[0:2] = decoded_pos # sync kf_state with task.
kf_model.set_state(kf_state) # VERY important!! Synchronize kf_model with kf_state
kf_state_cache_synced = kf_state.copy() # Cache the synced state in order to calculate the effective vel.

kf_obs = obs_arr[0:obs_dim] # Get the observation. shape (obs_dim,) == (hdim,)

# Calculate inferred state, then append to queue.
if sum(np.isnan(target_pos)) == 0: # some target has no position i.e still. sends np.nan as target pos

    inferred_state = gen_inferred_state(kf_state[0:2], target_pos, **{'decoded_vel': signed_vel, 'st_task': state_task})
    # remove position information if you want a velocity-only KF
    if inferred_state is not None:
        inferred_state[0:2] = 0.0
    inf_state_queue.append(inferred_state)
    if len(inf_state_queue) > queue_length:
        inf_state = inf_state_queue.popleft()
        kf_inf_state[:] = inf_state
    else:
        inf_state = None
        kf_inf_state[:] = np.nan

# Generate the next state
new_state = kf_model.step(kf_obs.copy()).flatten()
##new_state[0:2] = np.clip(new_state[0:2], -1, 1) # when uncommented: clip to the bounding box. Commented: leave up to other modules.
# prevent velocities from exploding by clipping l1 norm
l1 = np.sum(np.abs(new_state[2:6]))
if l1 > 1.0:
    new_state[2:6] = new_state[2:6]/l1
# place the new_state into shared memory kf_state
kf_state[:] = new_state # update the kf_state shared memory.


try:
    kf_ole_rlud[:] = kf_model.get_OLE_RLUD(kf_obs.copy()).flatten()
except:
    pass

if allow_kf_sync.item():
    # For now, ignore continuous_update for sanity.
    # Update the decoding parameters only if not in active state.
    kf_model.update_M1M2(verbosity=1)
    kf_state[2:6] = 0.0 # set velocities to 0
    kf_update_flag[:] = 1 # Communicate that kf M1&M2 is updated. Value 1

    save_partial = True
    if save_partial:
        kf_model.save(path=os.path.abspath(params['data_folder']) + '/partial_kf.npz')
    pass

elif allow_kf_adapt.item():
    # Update the Kalman Filter using the inferred state.
    # This does not update the decoding parameters M1 and M2.

    #if True:
    #    # remove position information if you want a velocity-only KF
    #    inferred_state[0:2] = 0.0
    if inf_state is not None:
        if refit_mode == 'split':
            if (inf_state[2:6] != 0).sum() == 1:
                kf_model.process_state_obs(inf_state, kf_obs.copy(), iterate_inv=True, kf_iter=1)
            else:
                inf_state_cache = inf_state.copy()
                selection = np.nonzero(inf_state[2:6] != 0)[0] + 2
                inf_state_list = [gen_default_state() for sel in selection]
                for state_i, sel in zip(inf_state_list, selection):
                    state_i[sel] = 1.0
                inf_state = np.stack(inf_state_list, axis=0)
                kf_model.process_state_obs(inf_state, kf_obs.copy()[None, :].repeat(inf_state.shape[0], axis=0), iterate_inv=True, kf_iter=1, sample_weights=inf_state_cache[selection])
        elif refit_mode == 'vector':
            kf_model.process_state_obs(inf_state, kf_obs.copy(), iterate_inv=True, kf_iter=1)
        else:
            raise ValueError('Invalid value of refit_mode!')
        kf_update_flag[:] = 2 # Communicate that kf M1&M2 is not updated, but kf_model is updated.
    else:
        kf_update_flag[:] = 0 # Communicate that kf M1&M2, nor kf_model is updated



# log kf parameters to shared memory
namespace = globals()
var_info = [
    (kf_model.R, 'kf_R'),
    (kf_model.S, 'kf_S'),
    (kf_model.T, 'kf_T'),
    (kf_model.Tinv, 'kf_Tinv'),
    (kf_model.EBS, 'kf_EBS'),
    (kf_model.C, 'kf_C'),
    (kf_model.Q, 'kf_Q'),
    (kf_model.Qinv, 'kf_Qinv'),
    (kf_model.S_k, 'kf_S_k'),
    (kf_model.K_k, 'kf_K_k'),
    (kf_model.M1, 'kf_M1'),
    (kf_model.M2, 'kf_M2'),
    (kf_state[0:2] - kf_state_cache_synced[0:2], 'kf_effective_vel'),
]
for info in var_info:
    v_self = info[0]
    try:
        v = namespace[info[1]]
        if isinstance(v_self, str):
            v[0:len(v_self)] = np.frombuffer(v_self.encode(), dtype='int8')
        else:
            v[:] = v_self
    except Exception as e:
        print(f'failed to save {info[1]} to sharedmemory', e)
# end logging
