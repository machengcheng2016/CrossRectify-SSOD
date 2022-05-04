import torch
from copy import deepcopy
from collections import OrderedDict

things1 = torch.load("mine3-unbiased-teacher-VOC07-sup-bs16/output/model_0005999.pth")
things2 = torch.load("mine3-unbiased-teacher-VOC07-sup-bs16.2/output/model_0005999.pth")
model1, optimizer1 = things1['model'], things1['optimizer']
model2, optimizer2 = things2['model'], things2['optimizer']

things = deepcopy(things1)

# prepare model
model = OrderedDict()
for key, value in model1.items():
    words = key.split('.')
    words[1] = words[1] + '1'
    new_key = ".".join(words)
    model[new_key] = value
for key, value in model2.items():
    words = key.split('.')
    words[1] = words[1] + '2'
    new_key = ".".join(words)
    model[new_key] = value
things['model'] = model

# prepare optimizer, which contains state and param_groups
optimizer = dict()
state = dict()
state1, state2 = optimizer1['state'], optimizer2['state']
for i in range(58):
    state[i] = state1[i]
for i in range(58, 116):
    state[i] = state2[i-58]
for i in range(116, 122):
    state[i] = state1[i-58]
for i in range(122, 128):
    state[i] = state2[i-64]
for i in range(128, 136):
    state[i] = state1[i-64]
for i in range(136, 144):
    state[i] = state2[i-72]

param_groups = list()
param_groups1, param_groups2 = optimizer1['param_groups'], optimizer2['param_groups']
for i in range(58):
    param_groups.append(param_groups1[i])
for i in range(58, 116):
    param_groups.append(param_groups2[i-58])
for i in range(116, 122):
    param_groups.append(param_groups1[i-58])
for i in range(122, 128):
    param_groups.append(param_groups2[i-64])
for i in range(128, 136):
    param_groups.append(param_groups1[i-64])
for i in range(136, 144):
    param_groups.append(param_groups2[i-72])

optimizer['state'] = state
optimizer['param_groups'] = param_groups

# prepare scheduler
things['scheduler']['base_lrs'] += things2['scheduler']['base_lrs']
things['scheduler']['_last_lr'] += things2['scheduler']['_last_lr']

# dumping
things['optimizer'] = optimizer
for i in range(len(things['optimizer']['state'])):
    if 'momentum_buffer' in things['optimizer']['state'][i]:
        del(things['optimizer']['state'][i]['momentum_buffer'])

torch.save(things, "model_0005999.cotraining2.pth")
