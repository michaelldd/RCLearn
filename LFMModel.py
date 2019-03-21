# coding:utf-8
import math
import random

class LFMModel:

    '''
    根据user-item信息以及隐隐含因子数量初始化PQ矩阵
    输入：
      users：所有的用户
      items：所有的物品
      F:兴趣因子数
    '''
    def __init__(self, users, items,F):
        self.P = {}
        self.Q = {}
        self.user_items = {}
        self.mse_result = []
        self.F = F
        self.users = users
        self.items = items
        a = 1
        for user in users:
            self.P[user] = [random.random() for i in range(F)]
        for item in items:
            self.Q[item] = [random.random() for i in range(F)]

    '''
    Predict 计算预测值与真实值的误差
    输入：
        user：用户标识
        item：物品标识
        P
        Q
    返回：
        误差值
    '''
    def predict(self,user, item):
        return sum(self.P[user][f]*self.Q[item][f] for f in range(self.F))
    def recommend(self,user,item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F))
    '''
        计算单个预测评分的mse
        输入：
            user:用户表示
            item:物品标识
    '''
    def elem_mse(self,user,item):
        return math.pow(self.user_items[user][item] - sum(self.P[user][f]*self.Q[item][f] for f in range(self.F)),2)
    '''
        计算整个矩阵的mse
    '''
    def mse(self):
        scores = []
        for user in self.users:
            for item in self.items:
                scores.append(math.pow(self.user_items[user][item] - sum(self.P[user][f]*self.Q[item][f] for f in range(self.F)),2))
        return sum(scores)/len(scores)
        
    '''
    输入：
        user_items：（user-items字典）
        users: 包含所有user的列表
        items： 包含所有item的列表
        N：SGD迭代数
        alpha：alpha
        _lambda:正则化系数
    '''
    def train(self,user_items,steps,alpha,_lambda):
        self.user_items = user_items
        # 去除可能存在的未评分元素
        pop_elems = []
        new_user_items = {}
        for user,items in new_user_items.items():
            users = {}
            for item,rui in items.items():
                users[item] = rui
                if(new_user_items[user][item]==0):
                    pop_elems.append((user,item))
            new_user_items[user] = users
        
        for user,items in pop_elems:
            new_user_items[user].pop(item)
        
        
        #更新参数
        for step in range(steps):
            for user,items in new_user_items.items():
                for item,rui in items.items():
                    eui = rui - self.predict(user,item)
                    for f in range(self.F):
                        self.P[user][f] +=alpha * (eui * self.Q[item][f] - _lambda * self.P[user][f])
                        self.Q[item][f] +=alpha *(eui * self.P[user][f] - _lambda * self.Q[item][f])
            if(step % 10 ==0):
                self.mse_result.append(self.mse())
'''
user_items = {1: {'a': 1, 'b': 1, 'c': 1, 'd': 2, 'e': 2, 'f': 2, 'g': 3},
              2: {'a': 4, 'b': 4, 'c': 5, 'd': 5, 'e': 4, 'f': 3, 'g': 1},
              3: {'a': 3, 'b': 3, 'c': 3, 'd': 3, 'e': 3, 'f': 3, 'g': 4},
              4: {'a': 3, 'b': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
              5: {'a': 2, 'b': 3, 'c': 3, 'd': 3, 'e': 4, 'f': 3, 'g': 2},
              6: {'a': 3, 'b': 0, 'c': 3, 'd': 4, 'e': 3, 'f': 0, 'g': 2}}

# 我们要得到里面的users和items
users = {1, 2, 3, 4, 5, 6}
items = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

model = LFMModel(users,items,6)
model.train(user_items,F=10,alhpa=0.1,_lambda=0.1)
model.mse()
'''