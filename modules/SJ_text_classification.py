
# must be np array to keep variable's address
# in:
#   - state_task
#   - decoder_output
#   - target_hit
# out:
#   - decoded_pos
#   - target_hit
#   - scores




# l = decoder_output[0]
# r = decoder_output[1]
# u = decoder_output[2]
# d = decoder_output[3]
# s = decoder_output[4]

# decoder_output[0] = r
# decoder_output[1] = l
# decoder_output[2] = d
# decoder_output[3] = u



param = [decoder_output]
result = task.update(param)

decoded_pos[:]       = result[0]
state_task[:]        = result[1]
target               = result[2]
#print(state_task,target,decoded_pos,eegbufferindex)

try:
    text_pos[:] = task.text_pos
    text_direction[:] = {'L': 0, 'R': 1, 'U': 2, 'D': 3}[task.text_direction]
except Exception as e:
    #print(e)
    pass