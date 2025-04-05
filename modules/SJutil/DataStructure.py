import numpy as np
import collections.abc

""" CircularQueue for history """
class CircularQueue():
  def __init__(self,dim,obs_size):
    self.n_history = dim[0]
    self.interval = dim[1]
    self.obs_size = obs_size

    self.data_dim = ((self.n_history-1) * self.interval + 1, obs_size)
    self.n = self.data_dim[0]
    self.reset()

  def add(self,v):
    self.i += 1
    if self.i == self.n: self.i = 0
    self.data[self.i] = v

  def get(self):
    # return n history with interval
    selected = self.i - np.arange(self.n_history) * self.interval 
    return self.data[selected]
  
  def add_get(self,v):
    self.add(v)
    return self.get()
  
  def reset(self,pos = None):
    self.data = np.zeros(self.data_dim)
    self.i = 0
    if pos is not None:
      self.data[:,0:2] = pos

  def resetAllSoftmax(self):
    self.data[:,2:7] = 0

def deepDictUpdate(d1,d2):
  """Recursively update a dict.
  d1 is updated with d2 information
  """
  if not isinstance(d1, collections.abc.Mapping):
      return d2
  for key, value in d2.items():
      if isinstance(value, collections.abc.Mapping):
          d1[key] = deepDictUpdate(d1.get(key, {}), value)
      else:
          d1[key] = value
  return d1

if __name__ == "__main__":
  # example of deep dictionary copy
  d1 = { # first dictionary
    1:3,
    2:{
      4:5,
      'e':5,
      6:{
          '4':'4',
          '5':'a'
        },
    },
    3:4
  }
  d2 = { # second dictionary
    2:{
      6:{'4':'b'},
    }
  }
  deepDictUpdate(d1[2],d2[2])
  print(d1)
