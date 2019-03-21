
# coding:utf-8
import math
import random
import numpy as np

class SVDPlsuPlus():
    '''
    属性：
        P、Q：P、Q矩阵
        bu：user偏置项
        bi：item偏置项
        mu：总体平均分
        F：隐式因子数
        y: 隐式反馈矩阵
        user_items:数据字典
        users：包含所有用户的列表
        items：包含所有字段的列表
        mse_result:每十次迭代保存一次mse
    '''
    '''
        输入：
            F：隐式因子数
            users：包含所有user的列表
            items：包含所有item的列表
    '''
    def __init__(self,F,users,items):
        self.P = {}
        self.Q = {}
        self.mu=0
        self.bu = 0
        self.bi = 0
        self.user_items = {}
        self.mse_result=[]
        self.y = {}

        self.F = F
        self.users = users
        self.items = items
        for user in users:
            self.P[user] = [random.random() for i in range(F)]
        for item in items:
            self.Q[item] = [random.random() for i in range(F)]
            self.y[item] = [random.random() for i in range(F)]
    
    '''
        热门item,暂无用处
        输入：
            user_items矩阵
        返回：
            item_pool-->排序的物品列表（越火热的越靠前）
    '''
    def sort_items(self,user_items):
        item_pool = {}
        for user,items in user_items:
            for item,rui in items:
                if item not in item_pool:
                    item_pool[item] = rui
                else:
                    item_pool[item] += rui
        return item_pool.keys()
    
    
    '''
      得到公式中关于y的部分：|y|^2，∑yj/|y|^2  
      输入：
          user：用户标识
      返回：
          nu_sqrt,nu_tmp
    '''
    def get_y(self,user):
        nu_len = len(user_items[user])
        nu_sqrt = math.pow(nu_len,2)
        nu_tmp = np.sum([self.y[item] for item,rui in user_items[user].items()],axis=0)/nu_sqrt
        return nu_sqrt,nu_tmp
    '''
        预测 mu + bi + bu + qi的转置 * (pu + nu_tmp)
        输入：
            user:用户标识
            item:物品标识
        返回：
            rui：预测分数
    '''
    def predict(self,user,item):
        nu_sqrt,nu_tmp = self.get_y(user)
        return self.mu+self.bu[user]+self.bi[item]+np.sum(
            (np.array(self.P[user])+np.array(nu_tmp))*(np.array(self.Q[item]))
        )
    def recommend(self,user,item):
        nu_sqrt,nu_tmp = self.get_y(user)
        return self.mu+self.bu[user]+self.bi[item]+np.sum(
            (np.array(self.P[user])+np.array(nu_tmp))*(np.array(self.Q[item]))
        )
    '''
        输入：
            user_items: 字典类型数据
            alpha：alpha
            _kambda:正则化项参数
    '''
    def train(self,user_items,steps,alpha,_lambda):
        # 去除输入数据中可能存在的未评分元素
        self.user_items = user_items
        pop_elems = []
        new_user_items = {}
        for user,items in user_items.items():
            users = {}
            for item,rui in items.items():
                users[item] = rui
                if(user_items[user][item]==0):
                    pop_elems.append((user,item))
            new_user_items[user] = users
        for user,item in pop_elems:
            new_user_items[user].pop(item)
        
        #初始化mu、bi、bu
        user_ruis = {}
        item_ruis = {}
        for user,items in new_user_items.items():
            for item,rui in items.items():
                if(item not in item_ruis):
                    item_ruis[item] = [rui]
                else:
                    item_ruis[item].append(rui)
                if(user not in user_ruis):
                    user_ruis[user] = [rui]
                else:
                    user_ruis[user].append(rui)

        self.bu = {user:np.mean(ruis) for user,ruis in user_ruis.items()}
        self.bi = {item:np.mean(ruis) for item,ruis in item_ruis.items()}
        self.mu = np.mean([ruis for item,ruis in self.bu.items()])
        
        
        # 更新参数
        for step in range(steps):
            for user,items in user_items.items():
                for item,rui in items.items():
                    eui = rui - self.predict(user,item)
                    self.bu[user] += alpha*(eui-_lambda*self.bu[user])
                    self.bi[item] +=alpha*(eui-_lambda*self.bi[item])
                    nu_sqrt,nu_tmp = self.get_y(user)
                    for f in range(self.F):
                        self.P[user][f] +=alpha * (eui * self.Q[item][f] - _lambda * self.P[user][f])
                        self.Q[item][f] +=alpha *(eui * (self.P[user][f]+nu_tmp) - _lambda * self.Q[item][f])
                        for item_02,rui_02 in items.items():
                            self.y[item_02][f] +=alpha*(eui - nu_sqrt - _lambda*self.y[item_02][f]) 
            if(step % 10 ==0):
                self.mse_result.append(self.mse())
            #alpha = alpha*0.9
    
    '''
        计算单个预测评分的mse
        输入：
            user:用户表示
            item:物品标识
    '''
    def elem_mse(self,user,item):
        return math.pow(self.user_items[user][item] - self.predict(user,item),2)
    '''
        计算整个矩阵的mse
    '''
    def mse(self):
        scores = []
        for user in self.users:
            for item in self.items:
                scores.append(math.pow(self.user_items[user][item] - self.predict(user,item),2))
        return sum(scores)/len(scores)

'''
random.seed(2019)

user_items = {1: {'a': 1, 'b': 1, 'c': 1, 'd': 2, 'e': 2, 'f': 2, 'g': 3},
              2: {'a': 4, 'b': 4, 'c': 5, 'd': 5, 'e': 4, 'f': 3, 'g': 1},
              3: {'a': 3, 'b': 3, 'c': 3, 'd': 3, 'e': 3, 'f': 3, 'g': 4},
              4: {'a': 3, 'b': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
              5: {'a': 2, 'b': 3, 'c': 3, 'd': 3, 'e': 4, 'f': 3, 'g': 2},
              6: {'a': 3, 'b': 0, 'c': 3, 'd': 4, 'e': 3, 'f': 0, 'g': 2}}

# 我们要得到里面的users和items
users = {1, 2, 3, 4, 5, 6}
items = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

model = SVDPlsuPlus(F=5,users=users,items=items)
model.train(user_items,steps=100,alpha=0.0005,_lambda=0.1)

import seaborn as sns

sns.pointplot(x=list(range(10,len(model.mse_result)*10+1,10)),y=model.mse_result)
'''