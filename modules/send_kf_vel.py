
current_kf_state = kf_state.copy()
vel = kf_state_to_vel(current_kf_state, normalize='l1', clip=False, alpha=0.2)
kf_sent_vel[:] = vel # store the nonrotated and nonflipped value.

if flipx:
    vel[0] = (-1)*vel[0]
if flipy:
    vel[1] = (-1)*vel[1]
vel = rotation_matrix@vel
#
msg = vel.tobytes()
conn_retval = conn.sendall(msg)
print(conn_retval, vel)
