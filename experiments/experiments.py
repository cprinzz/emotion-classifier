import json
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

e1 = {'model_id':1,
      'X_train_fname':'../data/X_train.npy',
      'Y_train_fname':'../data/Y_train.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':10,
     }

e2 = {'model_id':2,
      'X_train_fname':'../data/X_train.npy',
      'Y_train_fname':'../data/Y_train.npy',
      'conv_layers':[{'depth':32,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                    {'units':128, 'activation':'relu'},
                    {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':10,
     }

e3 = {'model_id':3,
      'X_train_fname':'../data/X_train.npy',
      'Y_train_fname':'../data/Y_train.npy',
      'conv_layers':[{'depth':32,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':64, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':10,
     }

e4 = {'model_id':4,
      'X_train_fname':'../data/X_train.npy',
      'Y_train_fname':'../data/Y_train.npy',
      'conv_layers':[{'depth':32,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':64, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':10,
     }

e5 = {'model_id':5,
      'X_train_fname':'../data/X_train_192.npy',
      'Y_train_fname':'../data/Y_train_192.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':15,
     }

e6 = {'model_id':6,
      'X_train_fname':'../data/X_train_192.npy',
      'Y_train_fname':'../data/Y_train_192.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':64, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':15,
     }

e7 = {'model_id':7,
      'X_train_fname':'../data/X_train_192.npy',
      'Y_train_fname':'../data/Y_train_192.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.5,
      'epochs':15,
     }

e8 = {'model_id':8,
      'X_train_fname':'../data/X_train_scaled_jaffe.npy',
      'Y_train_fname':'../data/Y_train_scaled_jaffe.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.25,
      'epochs':15,
     }

e9 = {'model_id':9,
      'X_train_fname':'../data/X_train_scaled_jaffe.npy',
      'Y_train_fname':'../data/Y_train_scaled_jaffe.npy',
      'conv_layers':[{'depth':32,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.25,
      'epochs':15,
     }

e10 = {'model_id':10,
      'X_train_fname':'../data/X_train_scaled_jaffe_ck_dlib.npy',
      'Y_train_fname':'../data/Y_train_scaled_jaffe_ck_dlib.npy',
      'conv_layers':[{'depth':32,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.25,
      'epochs':15,
     }

e11 = {'model_id':11,
      'X_train_fname':'../data/X_train_scaled_jaffe_ck_dlib.npy',
      'Y_train_fname':'../data/Y_train_scaled_jaffe_ck_dlib.npy',
      'conv_layers':[{'depth':32,'filter':(5,5), 'activation':'relu', 'num':3},
                     {'depth':64,'filter':(3,3), 'activation':'relu', 'num':3},
                     {'depth':128,'filter':(3,3), 'activation':'relu', 'num':3}],
      'dense_layers':[{'units':128, 'activation':'relu'},
                      {'units':64, 'activation':'relu'},
                      {'units':6, 'activation':'softmax'}],
      'dropout':.25,
      'epochs':15,
     }

experiments = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]

with open('experiments.json', 'w') as f:
    f.write(json.dumps(experiments))

e_dict = {'model_id':[],
          'X_train_fname':[],
          'Y_train_fname':[],
          'dropout':[],
          'epochs':[],
          'acc':[],
          'val_acc':[],
          'loss':[],
          'val_loss':[]
         }

# for e in experiments:
#     e_dict['model_id'].append(e['model_id'])
#     e_dict['X_train_fname'].append(e['X_train_fname'])
#     e_dict['Y_train_fname'].append(e['Y_train_fname'])
#     e_dict['dropout'].append(e['dropout'])
#     e_dict['epochs'].append(e['epochs'])
#     e_dict['acc'].append(np.nan)
#     e_dict['val_acc'].append(np.nan)
#     e_dict['loss'].append(np.nan)
#     e_dict['val_loss'].append(np.nan)

df = DataFrame.from_dict(e_dict)
print(pd.read_csv('results.csv')[list(e_dict.keys())])
