
from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import wandb
import signal
import numpy as np

# NOTE: had to make global var accesed by callback bc otherwise cannot save model (TypeError: cannot pickle 'LazyModule' object). this means there should only be one copy of  LearningRateCallback
_STORED_LR = 0
def substitue_lr_schedule(x):
    return _STORED_LR

class LearningRateCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_callback, n_update_per_log, lrSchedulerInfo, total_timesteps, wandbUsed=False, verbose=1):
        super(LearningRateCallback, self).__init__(verbose=0)
        self.eval_callback = eval_callback
        self.n_rolloutPhase = n_update_per_log # number of update performed before logging
        self.rolloutPhase = 0 # phase goes from 1-n, at n it should do rl scheduling
        self.total_timesteps = total_timesteps
        self.verbose = verbose

        self.wandbUsed = wandbUsed

        self.lrSchedulerInfo = lrSchedulerInfo
        self.lr_min = lrSchedulerInfo["lr_min"]
        self.lr = lrSchedulerInfo["lr"]
        self.lr_max = lrSchedulerInfo["lr"]
        self.lr_scheduler = lrSchedulerInfo["lr_scheduler"]
        self.lr_patience = lrSchedulerInfo["lr_patience"]

        print("Learning Rate Callback Info:")
        # "constant / reducelronplateau / linear / exp"
        if self.lr_scheduler == "constant": self.retrieveLR = self.getConstantLR
        elif self.lr_scheduler == "reducelronplateau": self.retrieveLR = self.getReduceOnPlateuLR
        elif self.lr_scheduler == "linear": self.retrieveLR = self.getLinearLR
        elif self.lr_scheduler == "exp": self.retrieveLR = self.getExpLR
        else: 
            print("Error: type [{self.lr_scheduler}] lr_scheduler does not exist!")
            exit(1)
        print(f"Using Learning Rate Scheduler: [{self.lr_scheduler}]")
        print()

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        global _STORED_LR
        """
        This method is called before the first rollout starts.
        """

        # scheduler created
        optimizer = self.model.policy.optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, 'max', patience=self.lr_patience, verbose=True)

        # use substitue
        _STORED_LR = self.model.lr_schedule(1)
        self.model.lr_schedule = substitue_lr_schedule


    # "constant / reducelronplateau / linear / exp"
    def getConstantLR(self): 
        
        return self.lr

    def getReduceOnPlateuLR(self): 

        # update lr with scheduler
        last_mean_reward = self.eval_callback.last_mean_reward
        self.scheduler.step(last_mean_reward)

        # retrieve the updated lr
        for param_group in self.model.policy.optimizer.param_groups:
            lr = param_group["lr"]
        
        if self.verbose: print("scheduler stepped with (val reward,lr):",last_mean_reward,self.lr)
        # if self.wandbUsed: wandb.log({'log lr': self.lr})

        # close program if lr is too small
        if lr < self.lr_min:
            print("lr is too small for legitimate training",lr)
            signal.raise_signal( signal.SIGINT )

        return lr
            
    def getLinearLR(self): 
        progress = self.num_timesteps / self.total_timesteps
        lr = self.lr_max - (self.lr_max - self.lr_min) * progress
        if self.verbose: print(f"scheduler stepped with (completion, lr): {progress*100:.2f}% , {lr}")
        return lr

    def getIncorrectLinearLR(self): 
        progress = self.num_timesteps / self.total_timesteps
        lr = self.lr - (self.lr - self.lr_min) * progress
        if self.verbose: print(f"scheduler stepped with (completion, lr): {progress*100:.2f}% , {lr}")
        return lr

    def getExpLR(self):
        # not implemented yet!
        return self.lr

    def _on_rollout_end(self) -> None:
        global _STORED_LR
        """
        This event is triggered before updating the policy.
        """

        # runs only once in self.n_rolloutPhase 
        if self.rolloutPhase < self.n_rolloutPhase-1:
            self.rolloutPhase += 1
            return
        self.rolloutPhase = 0

        # set lr 
        _STORED_LR = self.lr = self.retrieveLR()


        
    def print_optimizer_info(self):

        print("----- PRINTING OPTIMIZER INFO -----")
        for param_group in self.model.policy.optimizer.param_groups:
            for k,v in param_group.items():
                if k != "params":
                    print("optimizer",k,v)
        print()

    # unused callbacks but maybe for the future


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass



import numpy as np

class TensorboardLoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0,env=None,eval_env=None,n_update_per_log=1, n_eval_episodes=4):
        super(TensorboardLoggerCallback, self).__init__(verbose)
        self.env = env
        self.eval_env = eval_env
        self.value = 0
        self.n_update_per_log = n_update_per_log
        self.n_eval_episodes = n_eval_episodes

        # hyper parameter constants
        self.tickLength = self.eval_env.taskGame.tickLength
        for target_name, (target_pos, target_size) in self.eval_env.taskGame.targetsInfo.items():
            if target_name == 'center': continue
            self.straight_distance = np.linalg.norm(target_pos)
            self.session_target_diameter = target_size[0]
            break
        # fixed for apple to apple evaluation purpose
        self.tickLength = 0.05 # 50ms
        self.straight_distance = 0.7 # assume that the cursor starts from the center. This is not always the case however
        self.session_target_diameter = 0.2


        self.update_counter = 0
        self.previousSuccessIndex = 0

    def _on_rollout_end(self) -> None:
        # this is where we log anything we want
        if self.update_counter == self.n_update_per_log:
                    
            # log eval success out of 4 trial that was ran
            # print('eval_success',self.eval_env.trialResults[-4:])
            evalTrialSuccesses = np.array(self.eval_env.trialResults[-self.n_eval_episodes:]) > 0 # success of last n trial
            evalTrialTicks = np.array(self.eval_env.taskGame.PerformanceRecord.timeToHitByTrials[-self.n_eval_episodes:])
            distancesTraveled = np.array(self.eval_env.taskGame.PerformanceRecord.distanceTravelledByTrials[-self.n_eval_episodes:])
            extraDistancesTraveled = np.array(self.eval_env.taskGame.PerformanceRecord.extraDistanceTravelledByTrials[-self.n_eval_episodes:])
                
            if self.eval_env.taskGame.centerIn: # uses center in. only count center out
                evalTrialSuccesses = evalTrialSuccesses[::2]
                evalTrialTicks = evalTrialTicks[::2]
                distancesTraveled = distancesTraveled[::2]
                extraDistancesTraveled = extraDistancesTraveled[::2]
            
            evalEndSuccess = sum(evalTrialSuccesses)
            evalSuccessPercentage = sum(evalTrialSuccesses) / len(evalTrialSuccesses)
            
            avgEvalTrialTime = np.mean(evalTrialTicks) * self.tickLength
            avgDistanceTraveled = np.mean(distancesTraveled)
            avgExtraDistanceTraveled = np.mean(extraDistancesTraveled)
            avgPathEfficiency = np.mean((self.straight_distance / distancesTraveled) * 100.0)
            hitRate = evalEndSuccess / (sum(evalTrialTicks) * self.tickLength)

            iod = np.log2(1 + self.straight_distance/self.session_target_diameter)
            itr = iod / avgEvalTrialTime
            

            # trialTime =(np.array(self.eval_env.trialResults[-self.n_eval_episodes:]) > 0).sum() / self.n_eval_episodes # success in percentage

            self.logger.record('eval/n_success', evalEndSuccess)
            self.logger.record('eval/success_percentage', evalSuccessPercentage)
            self.logger.record('eval/trial_time', avgEvalTrialTime)
            self.logger.record('eval/distance_traveled', avgDistanceTraveled)
            self.logger.record('eval/extra_distance_traveled', avgExtraDistanceTraveled)
            self.logger.record('eval/path_efficiency', avgPathEfficiency)
            self.logger.record('eval/hit_rate', hitRate)
            self.logger.record('eval/fitt_ITR', itr)
            
            print('eval/n_success', evalEndSuccess)
            print('eval/success_percentage', evalSuccessPercentage)
            print('eval/trial_time', avgEvalTrialTime)
            print('eval/distance_traveled', avgDistanceTraveled)
            print('eval/extra_distance_traveled', avgExtraDistanceTraveled)
            print('eval/path_efficiency', avgPathEfficiency)
            print('eval/hit_rate', hitRate)
            print('eval/fitt_ITR', itr)
            

            print('eval_env',len(self.eval_env.trialResults))
            print('env',len(self.env.trialResults))

            # log success of out of total number of trials that was ran within this tiem frame
            totalTrials = np.clip((len(self.env.trialResults)-self.previousSuccessIndex),1,np.inf)
            successTrials = np.sum(np.array(self.env.trialResults[self.previousSuccessIndex:]) > 0)
            trialEndSuccess = successTrials / totalTrials
            # print(successTrials,totalTrials)
            # print('success',self.env.trialResults[self.previousSuccessIndex:],trialEndSuccess)
            # print('trial',self.env.trialResults)
            self.logger.record('success', trialEndSuccess)
            self.previousSuccessIndex = len(self.env.trialResults)

            # log last 10 success
            tenSuccess = (np.array(self.env.trialResults[-10:]) > 0).sum() # success of last 10 trial
            self.logger.record('last_ten_success', tenSuccess)


            # record curriculum if any are used
            if self.env.use_curriculum:
                for attribute, rule in self.env.curriculum.items():
                    self.logger.record(attribute, rule["current"])


            # log targetsize and set targetSize and change targetsize for eval as well. 
            # 05/22/23 update: maybe shouldn't change targetsize for eval. does not give good representation of eval. since eval is not even changing weights anyways
            
            # targetSize = self.env.targetSize
            # self.logger.record('target_size',targetSize)
            # self.eval_env.targetSize = self.eval_env.taskGame.targetSize = targetSize
            # self.eval_env.taskGame.changeTargetSize(targetSize)

            # log eval target up down left right

            self.update_counter = 1
        else:
            self.update_counter += 1
        
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True