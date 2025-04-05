
import numpy as np

class KalmanFilter():
    def __init__(self, state_dim=7, obs_dim=2, dt=0.020, lam=0.999, gain=1.0):
        # state dim is 2 for position, 2 each for velocity components,
        # and 1 for constant 1.
        # currently, this is hardcoded for 2 + 2 (x vel) + 2 (y vel) + 1
        # dt: in seconds.
        # min_x: minimum number of samples needed until fitting KF from sufficient statistics
        n_v_dim = (state_dim - 3) // 4

        self.A = self.get_A(dt, gain)
        
        self.W = np.zeros((state_dim, state_dim))
        self.W[[2, 3, 4, 5], [2, 3, 4, 5]] = 1.0
        
        self._counter = 0
        self.lam = lam
        self.gain = gain # gain multiplier for Kalman adjustment

        gamma = 1.0
        # sufficient statistics
        # stores the full matrix
        self.R = np.zeros((state_dim, state_dim))
        self.S = np.zeros((obs_dim, state_dim))
        self.T = gamma*np.eye(obs_dim)
        self.Tinv = 1/gamma*np.eye(obs_dim)
        self.EBS = 0.0
        
        # set state estimate cov and Kalman gain.
        self.S_k = np.zeros((state_dim, state_dim)) # cov(x_{k}|{y}_{1}^{k})
        self.K_k = np.zeros((state_dim, obs_dim))
        
        self.state = np.zeros(state_dim)
        self.state[-1] = 1.0
        return
    def fit(self, states, obss, prev_states=None, max_iter=100, EBS=None):
        # states: shape (K, state_dim)
        # obss: shape (K, obs_dim)
        # prev_states: shape (K, state_dim) corresponding to the state before each obss (optional)
        #   Do not use prev_states if you don't want to fit A and W
        # EBS: set the effective batch size for this batch, i.e. rescale it.
        
        # This method uses whatever is stored in self.A and self.W if prev_states is None
        # Otherwise, it fits only the velocity diagonals of the A and W matrix
        
        states = states.astype('float64')
        obss = obss.astype('float64')

        K = states.shape[0]
        if prev_states is not None:
            # calculate full A and W and assign their diagonals to self.A and self.W
            # A
            prev_states = prev_states.astype('float64')
            vel_sel = [2, 3, 4, 5] # indices of velocity dimensions
            A = (states.T@prev_states)@np.linalg.inv(prev_states.T@prev_states) # commented out to use preset instead
            self.A[vel_sel, vel_sel] = A[vel_sel, vel_sel]
            # W
            ws = states - prev_states@self.A.T # shape (K, state_dim)
            W = 1/K*ws.T@ws # shape (state_dim, state_dim)
            self.W[vel_sel, vel_sel] = W[vel_sel, vel_sel]
        
        if EBS is None:
            EBS = K
        self.R = (EBS/K)*(states.T@states)
        self.S = (EBS/K)*(obss.T@states)
        self.T = (EBS/K)*(obss.T@obss)
        self.Tinv = np.linalg.inv(self.T)
        self.EBS = EBS
        
        self.C = self.S@np.linalg.pinv(self.R) # use pseudoinverse instead of inverse to account for constant position
        self.Q = 1/self.EBS*(self.T - self.C@self.S.T)
        self.Qinv = np.linalg.inv(self.Q)
        
        self.M1, self.M2, self.S_k, self.K_k = self.kf_recursion(self.A, self.W, self.C, Q=self.Q, max_iter=max_iter, pos_uncertainty=False)
        return self
    @staticmethod
    def get_A(dt, gain=1.0):
        # hard-code state_dim as 7
        # dt in seconds.
        A = np.zeros((7, 7))
        A[[0, 1], [0, 1]] = 1.0 # position
        A[[0, 0], [2, 3]] = [gain*dt, -gain*dt] # x pos update from vel
        A[[1, 1], [4, 5]] = [gain*dt, -gain*dt] # y pos update from vel
        A[-1, -1] = 1.0 # constant 1
        
        # Ref: Silversmith
        # for velocity components, we note that 0.825**(5 bins/second) = 0.382 /second
        # We thus set the velocity diagonal to a = (0.825**5)**(dt), when dt is in seconds,
        #   such that a**(1/dt) = (0.825**5)
        A[[2, 3, 4, 5], [2, 3, 4, 5]] = (0.825**5)**(dt)
        return A
    def set_W_diag(self, val):
        self.W[[2, 3, 4, 5], [2, 3, 4, 5]] = val
        return
    def rescale_EBS(self, EBS):
        if EBS is None:
            return
        self.R = (EBS/self.EBS)*self.R
        self.S = (EBS/self.EBS)*self.S
        self.T = (EBS/self.EBS)*self.T
        self.Tinv = (self.EBS/EBS)*self.Tinv
        self.EBS = EBS
        return
    def load(self, param_dict):
        self.R = param_dict['R'].astype('float64')
        self.S = param_dict['S'].astype('float64')
        self.T = param_dict['T'].astype('float64')
        self.Tinv = np.linalg.pinv(self.T)
        self.EBS = param_dict['EBS']
        self.S_k = param_dict['S_k'].astype('float64')
        self.K_k = param_dict['K_k'].astype('float64')
        self.A = param_dict['A'].astype('float64')
        self.W = param_dict['W'].astype('float64')
        # set self.C, self.Q, and self.Qinv based on the loaded R, S, T
        self.C = self.S@np.linalg.pinv(self.R) # use pseudoinverse instead of inverse to account for constant position
        self.C[:, 0:2] = 0.0 # i.e. use pure VKF
        self.Q = 1/self.EBS*(self.T - self.C@self.S.T)
        self.Qinv = np.linalg.inv(self.Q)
        return
    def save(self, path=None):
        param_dict = {
            'R': self.R,
            'S': self.S,
            'T': self.T,
            'EBS': self.EBS,
            'S_k': self.S_k,
            'K_k': self.K_k,
            'A': self.A,
            'W': self.W,
        }
        if path is not None:
            np.savez(path, **param_dict)
        return param_dict
    
    def process_state_obs(self, state, obs, iterate_inv=True, kf_iter=1, sample_weights=None):
        '''
        Updates self.S_k and self.K_k. Does not update M1 and M2.
        # state: shape (new_K, state_dim) or (state_dim,)
        # obs: shape (new_K, obs_dim) or (obs_dim,)
        # iterate_inv=True for one-step updates, =False for recomputed inverse
        # this function updates the sufficient statistics
        #   and updates the recursive parameters up to kf_iter number of times.
        #   this assumes that it receives the points in sequence and weights them accordingly using self.lam.
        '''
        
        # change: this function recurses exactly once.
        if state.ndim == 1:
            state = state.reshape((1, -1))
            obs = obs.reshape((1, -1))
        state = state.astype('float64')
        obs = obs.astype('float64')
        
        new_K = state.shape[0] # number of samples
        # assumed that state[0] and obs[0] is the furthest in the past
        #   and state[-1] and obs[-1] is the most recent
        if sample_weights is None:
            sample_weights = np.flip(self.lam**np.arange(new_K))
            past_weight = (self.lam**new_K)
        else:
            # assumes that the sample_weights all correspond to one sample.
            past_weight = self.lam
        
        # update sufficient statistics
        self.R = past_weight*self.R + (state.T*sample_weights)@state
        self.S = past_weight*self.S + (obs.T*sample_weights)@state
        self.T = past_weight*self.T + (obs.T*sample_weights)@obs
        
        # update Tinv, either by iterating (efficient when new_K is small) or recomputing.
        if iterate_inv:
            for obs_i in obs:
                obs_i = obs_i.reshape((-1, 1))
                Tinv_y = self.Tinv@obs_i # shape (obs_dim, 1)
                # note that T is symmetric so Tinv@y = y.T@Tinv
                numer = (1/(self.lam**2))*Tinv_y@Tinv_y.T
                denom = 1 + (1/self.lam)*obs_i.T@Tinv_y
                self.Tinv = (1/self.lam)*self.Tinv - numer/denom.item()
                pass
        else:
            # recompute Tinv from scratch
            self.Tinv = np.linalg.inv(self.T)
        
        self.EBS = past_weight*self.EBS + np.sum(sample_weights)
        
        # using pinv
        self.C = self.S@np.linalg.pinv(self.R)
        self.C[:, 0:2] = 0.0 # i.e. use pure VKF
        Tinv_S = self.Tinv@self.S
        self.Qinv = self.EBS*(self.Tinv - Tinv_S@np.linalg.pinv(-self.R + self.S.T@Tinv_S)@Tinv_S.T) # update Q_inv for use in sig update
        
        # update Q (not usually necessary)
        self.Q = 1/self.EBS*(self.T - self.C@self.S.T)
        
        # apply kalman filter recursion at most kf_iter number of times
        self.S_k, self.K_k = self.kf_iter(self.A, self.W, self.C, Qinv=self.Qinv, max_iter=kf_iter,
                                          S_c=self.S_k, K_c=self.K_k, pos_uncertainty=False)
        
        self._counter += new_K
        return
    def update_M1M2(self, verbosity=0):
        '''Synchronizes M1 and M2 (used for decoding) with Kalman gain K_k (not directly used for decoding)'''
        self.M1 = (self.A - self.K_k@self.C@self.A).copy()
        self.M2 = self.K_k.copy()
        if verbosity > 0:
            print('M1 M2 updated!')
        return
    def set_state(self, state_to_set):
        self.state[:] = state_to_set
        return
    def step(self, obs):
        '''decode the state from state and obs inputs using self.M1 and self.M2, and unrolling with self.state'''
        # state: shape (new_K, state_dim) or (state_dim,)
        # obs: shape (new_K, obs_dim) or (obs_dim,)
        # this is separate from process_state_obs in order to reduce cheating
        # returns shape (new_K, state_dim) of stepped states.
        if obs.ndim == 1:
            #state = state.reshape((1, -1))
            obs = obs.reshape((1, -1))
            
        states_out = []
        for obs_i in obs:
            self.state[:] = self.M1@self.state + self.M2@obs_i
            states_out.append(self.state.copy()) # use copy if you use in-place assignment.
        out = np.stack(states_out, axis=0)
        return out
    def get_OLE_RLUD(self, obs):
        '''Compute and return the least squares state corresponding to the current C.'''
        result = np.linalg.lstsq(self.C[:, [2, 3, 4, 5]], obs - self.C[:, 6], rcond=None)[0]
        result = result/np.maximum(np.abs(result).sum(), 1e-7)
        return result
    
    def update_C_Q(self):
        self.C = self.S@np.linalg.pinv(self.R)
        self.Q = 1/self.EBS*(self.T - self.C@self.S.T)
        return
    
    @staticmethod
    def get_ebs(n, lam):
        if lam == 1.0:
            result = 1.0*n
        else:
            result = 1.0*((1 - lam**(n-1))/(1 - lam))
        return result
    @staticmethod
    def kf_iter(A, W, C, Q=None, Qinv=None, 
                max_iter=100, tol=1e-10,
                K_c=None, S_c=None, pos_uncertainty=False):
        # Returns only S_c and K_c, so it is different from kf_recursion.
        S_c = S_c if S_c is not None else np.zeros_like(A) # current state estimate cov, i.e sigma_{k|k-1} and sigma_{k|k}. Initialized as sigma_{k|k}
        K_c = K_c if K_c is not None else np.zeros_like(C.T) # current Kalman gain
        
        # Use Qinv for speed.
        if Q is None and Qinv is None:
            raise ValueError('must provide either Q or Qinv!')
        if Qinv is None:
            Qinv = np.linalg.inv(Q)
        Ct_Qinv = C.T@Qinv
        for ii in range(max_iter):
            # cache the kalman gain for break criterion.
            K_p = K_c
            
            S_c = A@S_c@A.T + W # 1. Set sigma_{k|k-1}
            if not pos_uncertainty:
                S_c[0:2, :] = 0 # no uncertainty in position
                S_c[:, 0:2] = 0 # no uncertainty in position
            
            # order matters: 1-3-2
            # Follows from push-through.
            Sc_Ct_Qinv = S_c@Ct_Qinv
            K_c = np.linalg.inv(np.eye(A.shape[0]) + Sc_Ct_Qinv@C)@Sc_Ct_Qinv # 3.
            
            S_c = (np.eye(S_c.shape[0]) - K_c@C)@S_c # 2. S_c is now sigma_{k|k}
            
            crit = (np.max(np.abs(K_c - K_p)) < tol)
            if crit:
                break
        return S_c, K_c
    @staticmethod
    def kf_recursion(A, W, C, Q=None, Qinv=None, 
                     max_iter=100, tol=1e-10,
                     S_c=None, M1=None, M2=None,
                    pos_uncertainty=False):
        # to-do: init these:
        S_c = S_c if S_c is not None else np.zeros_like(A) # current state estimate cov, i.e sigma_{k|k-1} and sigma_{k|k}. Initialized as sigma_{k|k}
        M1_p = M1 if M1 is not None else np.zeros_like(A) # prev M1
        M1_c = M1 if M1 is not None else np.zeros_like(A) # curr M1
        M2_p = M2 if M2 is not None else np.zeros_like(C.T) # prev M2
        M2_c = M2 if M2 is not None else np.zeros_like(C.T) # curr M2
        
        # Use Qinv for speed.
        if Q is None and Qinv is None:
            raise ValueError('must provide either Q or Qinv!')
        if Qinv is None:
            Qinv = np.linalg.inv(Q)
        
        for ii in range(max_iter):
            M1_p = M1_c
            M2_p = M2_c
            S_c = A@S_c@A.T + W # 1. Set sigma_{k|k-1}
            if not pos_uncertainty:
                S_c[0:2, :] = 0 # no uncertainty in position
                S_c[:, 0:2] = 0 # no uncertainty in position
            
            # order matters: 1-3-2
            # Follows from push-through.
            Sc_Ct_Qinv = S_c@C.T@Qinv
            K_st = np.linalg.inv(np.eye(A.shape[0]) + Sc_Ct_Qinv@C)@Sc_Ct_Qinv # 3.
            
            S_c = (np.eye(S_c.shape[0]) - K_st@C)@S_c # 2. S_c is now sigma_{k|k}
            
            M1_c = A - K_st@C@A
            M2_c = K_st
            #
            
            crit = np.maximum(np.max(np.abs(M1_c - M1_p)), np.max(np.abs(M2_c - M2_p))) < tol
            if crit:
                break
        return M1_c, M2_c, S_c, K_st
