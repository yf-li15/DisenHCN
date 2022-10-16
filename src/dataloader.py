'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''
import os
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import world


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_scenes(self):
        raise NotImplementedError

    @property
    def n_apps(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def trainValue(self):
        raise NotImplementedError

    @property
    def trainDict(self):
        raise NotImplementedError

    @property
    def valDict(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allScenes(self):
        raise NotImplementedError
    
    

class Loader(BasicDataset):
    """
    data:(N, 5)
    user, base, time, app, value(freq for user preference)
    """
    def __init__(self, path=world.DATA_PATH):
        # load train or val or test data
        train_path = os.path.join(path, 'train_data.npz')
        trainmat = np.load(train_path)
        self.train_data = trainmat['train_data']
        val_path = os.path.join(path, 'val_data.npz')
        valmat = np.load(val_path)
        self.val_data = valmat['val_data']
        test_path = os.path.join(path, 'test_data.npz')
        testmat = np.load(test_path)
        self.test_data = testmat['test_data']

        self.traindataSize = len(self.train_data)
        self.scene_list = []
        self.n_app = 0

        # pre-calculate
        self.__trainDict, self.__valDict, self.__testDict = self.__build_test()
        self.__trainValue = self.get_trainValue()
        print("data loaded!")

    @property
    def n_scenes(self):
        return len(self.scene_list)

    @property
    def n_apps(self):
        return self.n_app
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def trainValue(self):
        return self.__trainValue

    @property
    def trainDict(self):
        return self.__trainDict
    
    @property
    def valDict(self):
        return self.__valDict
    
    @property
    def testDict(self):
        return self.__testDict
    
    @property
    def allScenes(self):
        return self.scene_list

    def get_trainValue(self):
        """
        return:
            dict: {(user,base, time, app) : value}
        """
        train_value = dict()
        for i in range(len(self.train_data)):
            u, b, t, a, v = int(self.train_data[i, 0]), int(self.train_data[i, 1]), int(self.train_data[i, 2]), int(self.train_data[i, 3]), float(self.train_data[i, 4])
            train_value[(u,b,t,a)] = v
        return train_value

    def __build_test(self):
        """
        return:
            dict: {(user,base, time) : [items]} or dict: {scene : [items]}
        """
        train_dict = dict()
        for i in range(len(self.train_data)):
            u, b, t, a = int(self.train_data[i, 0]), int(self.train_data[i, 1]), int(self.train_data[i, 2]), int(self.train_data[i, 3])
            self.n_app = max(self.n_app, a)
            if (u,b,t) not in train_dict.keys():
                train_dict[(u,b,t)] = [a]
                if (u,b,t) not in self.scene_list:
                    self.scene_list.append((u,b,t))
            else:
                train_dict[(u,b,t)].append(a)

        val_dict = dict()
        for i in range(len(self.val_data)):
            u, b, t, a = int(self.val_data[i, 0]), int(self.val_data[i, 1]), int(self.val_data[i, 2]), int(self.val_data[i, 3])
            self.n_app = max(self.n_app, a)
            if (u,b,t) not in val_dict.keys():
                val_dict[(u,b,t)] = [a]
                if (u,b,t) not in self.scene_list:
                    self.scene_list.append((u,b,t))
            else:
                val_dict[(u,b,t)].append(a)
        
        test_dict = dict()
        for i in range(len(self.test_data)):
            u, b, t, a = int(self.test_data[i, 0]), int(self.test_data[i, 1]), int(self.test_data[i, 2]), int(self.test_data[i, 3])
            self.n_app = max(self.n_app, a)
            if (u,b,t) not in test_dict.keys():
                test_dict[(u,b,t)] = [a]
                if (u,b,t) not in self.scene_list:
                    self.scene_list.append((u,b,t))
            else:
                test_dict[(u,b,t)].append(a)
        
        self.n_app += 1

        return train_dict, val_dict, test_dict

   

