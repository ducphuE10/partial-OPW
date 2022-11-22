import torch
import numpy as np
from os import path as osp
from models.model_utils import cosine_sim

def POT_feature_2sides(a,b,D, m=None, nb_dummies=1):
  # a = np.ones(D.shape[0])/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  if m < 0:
      raise ValueError("Problem infeasible. Parameter m should be greater"
                        " than 0.")
  elif m > np.min((np.sum(a), np.sum(b))):
      raise ValueError("Problem infeasible. Parameter m should lower or"
                        " equal than min(|a|_1, |b|_1).")
  # import ipdb; ipdb.set_trace()
  b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = np.zeros((len(a_extended), len(b_extended)))
  D_extended[-nb_dummies:, -nb_dummies:] = np.max(D) * 2
  D_extended[:len(a), :len(b)] = D
  return a_extended, b_extended, D_extended

def POT_feature_1side(a,b,D, m=0.7, nb_dummies=1):
  # a = np.ones(D.shape[0])*m/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  a = a*m
  '''drop on side b --> and dummpy point on side a'''
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = torch.zeros((len(a_extended), len(b)))
#   D_extended = F.pad(input=D, pad=(0, 0, 0, 1), mode='constant', value=0)
  D_extended[:len(a), :len(b)] = D
  return a_extended, b,D_extended

def opw_distance(D, lambda1=1, lambda2=0.1, delta=1):
  N =D.shape[0]
  M = D.shape[1]

  E = torch.zeros((N,M)).to(D.device)
  for i in range(N):
    for j in range(M):
      E[i,j] = 1/((i/N - j/M)**2 + 1)

  l = torch.zeros((N,M)).to(D.device)
  for i in range(N):
    for j in range(M):
      l[i,j] = abs(i/N - j/M)/(np.sqrt(1/N**2 + 1/M**2))
  F = l**2
  return D - lambda1*E + lambda2*(F/2 + np.log(delta*np.sqrt(2*np.pi)))

def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])

def compute_OT_costs(sample):
    step_features, frame_features = sample['step_features'], sample['frame_features']
    return 1 - cosine_sim(step_features, frame_features)


def compute_OPW_costs(D ,lambda1, lambda2, delta=1, m=None, dropBothSides = False):
    a = np.ones(D.shape[0])/D.shape[0]
    b = np.ones(D.shape[1])/D.shape[1]
    
    D = opw_distance(D, lambda1, lambda2, delta)
    if dropBothSides:
        a,b,D = POT_feature_2sides(a,b,D,m)
    else:
        #drop side b
        a,b,D = POT_feature_1side(a,b,D,m)

    a = torch.from_numpy(a).to(D.device)
    b = torch.from_numpy(b).to(D.device)
    return D,a,b