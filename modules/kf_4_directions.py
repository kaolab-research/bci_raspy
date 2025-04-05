
# must be np array to keep variable's address
# in:
#   - decoder_output
# out:
#   - decoded_pos
#   - target_pos
#   - target_size
#   - state_task


if task.ready(eegbufferindex[0]):

    # use kf_state as decoder_output
    if 'copilot_kf_state' in params:
        _kf_output = np.append(kf_state[[3,2,4,5]],np.zeros(1))
        _decoder_output = np.clip(_kf_output,0,1)
    else:
        _decoder_output = decoder_output


    # illegal override softmax (should not done in normal case)
    if task.overrideSyntheticSoftmax:
        if task.target is not None:
            _targetPos,_targetSize = task.targetsInfo[task.target]
        else:
            _targetPos = task.getTruthPredictorTargeterPos(task.target)
            _targetSize = task.defaultTargetSize
        if task.overrideSyntheticSoftmax == 'complex':
            decoder_output = softmax = task.tempSynSoft.complexSoftmax(task.cursorPos, _targetPos, _targetSize, CSvalue=1.0, stillCS=1.0)
        else: 
            decoder_output = softmax = task.tempSynSoft.twoPeakSoftmax(task.cursorPos, _targetPos, _targetSize, CSvalue=1.0, stillCS=1.0)
    

    # use "my" decoder_output if situation requires
    if task.controlSoftmax:
        _decoder_output = task.get_my_decoder_output(decoder_output)

    # prepare copilot output
    if copilotReady: 
        obs = task.get_env_obs(_decoder_output, action=specialOption)
        episode_start = False
        if task.activeTrialHasBegun:
            trial_per_episode_counter += 1
            if trial_per_episode_counter == trial_per_episode:
                trial_per_episode_counter = 0
                episode_start = True
                copilot_state = None
        copilot_output, copilot_state = copilot.predict(obs, deterministic=True, episode_start=episode_start, state=copilot_state)
        specialOption = "mask" if copilot_output[2] > 0.5 else None
    else: 
        copilot_output = None

    param = [_decoder_output, copilot_output, {}, kf_state]
    result = task.update_kf(param)

    decoded_pos[:]       = result[0]
    target_pos[:]        = result[1]
    try:
        target_size[:]       = result[2]
    except:
        # here when using polygon targets
        pass
    state_task[:]        = result[3]
    game_state[:]        = ord(result[4]) # 'single ascii value of task: choices ("h":holding target, "H":aquired target, "w":wrong holdng, "W":wrong acquisition, "T":time out. "n":not on a target [or neutral state])'
    allow_kf_adapt[:]    = result[6][0] # kf related variable
    allow_kf_sync[:]     = result[6][1]
    
    if task.gamify:
        click_signal[:] = task.manualClick
        click_alpha[:] = task.GamifyClass.mouseAlpha
        print(click_signal, click_alpha)
        decoder_output[:] = result[5]['softmax']
task.log_to_sharedmemory(globals())