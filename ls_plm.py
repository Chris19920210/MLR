# -*- coding: utf-8 -*-
import numpy as np
import function as fc
import copy
import pickle
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import random
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log',
                filemode='w')

class LSPLM:

    def __init__(self,
                 pieceNum = 12,
                 iterNum = 100,
                 intercept = True,
                 memoryNum = 10,
                 beta = 0.1,
                 lamb = 0.1,
                 terminate = True
                 ):
        """
        :param feaNum:  特征数
        :param classNum:    类别数
        :param iterNum:
        :param intercept:
        :param memoryNum:
        :param beta:
        :param lamb:
        :param u_stdev:
        :param w_stdev:
        """
        self.pieceNum = pieceNum
        self.iterNum = iterNum
        self.intercept = intercept
        self.memoryNum = memoryNum
        self.lossList = []
        self.beta = beta
        self.lamb = lamb
        self.aucList = []
        self.N = 0
        self.p = 0
        self._predict = np.vectorize(self._mlr)

    def fit(self,X, y):
        """
            训练ls-plm large scale piece-wise linear model
        :param data:
        :return:
        """

        # np.random.seed(0)
        N, p = X.shape
        self.N = N
        if self.intercept:
            self.p = p + 1
            pad = np.ones((N, p + 1))
            pad[:,:-1] = X
            X = pad
            del pad
        else:
            self.p = p

        it = 0
        ## Intialization
        weight_W = np.random.normal(0,0.1,(self.pieceNum, self.p))
        weight_U = np.random.normal(0,0.1,(self.pieceNum, self.p))
        sList_w = np.zeros((self.memoryNum, self.pieceNum, self.p))
        sList_u = np.zeros((self.memoryNum, self.pieceNum, self.p))
        yList_w = np.zeros((self.memoryNum, self.pieceNum, self.p))
        yList_u = np.zeros((self.memoryNum, self.pieceNum, self.p))
        loss = fc.calLoss(X, y, weight_W, weight_U, self.lamb, self.beta)
        GW, GU = fc.sumCalDerivative(weight_W, weight_U, X, y)
        LW, LU = fc.virtualGradient(weight_W, weight_U, GW, GU,self.beta,self.lamb)
        del GW,GU
        # print("loss: %s" % loss)
        # print("gradient_w: is")
        # for w in weight_W:
        #     print (w)
        # print("gradient_u: is")
        # for u in weight_U:
        #     print(u)
        # self.firstLoss = loss
        # self.lossList.append(loss)
        #
        while it < self.iterNum:
            logging.info('===========iterator : %s =============' % it)
            logging.info('===========loss : %s ==============' % loss)
            start_time = time.time()
            # 1. 计算虚梯度
                #计算梯度
            GW, GU = fc.sumCalDerivative(weight_W, weight_U, X, y)
            newLW, newLU = fc.virtualGradient(weight_W, weight_U, GW, GU,self.beta,self.lamb)
            del GW,GU


            # dirW = copy.deepcopy(vGW)
            # dirU = copy.deepcopy(vGU)

            # 3. 利用LBFGS算法的两个循环计算下降方向, 这里会直接修改vGradient, 并确定下降方向是否跨象限

            PW, PU = fc.lbfgs(newLW ,newLU, sList_w,sList_u, yList_w, yList_u, it, self.memoryNum, it % self.memoryNum)

            # # 4. 确定下降方向是否跨象限， 这里也会直接修改vGradient
            # fc.fixDirection(vG, dir)

            # 5. 线性搜索最优解
            new_weight_W, new_weight_U = fc.backTrackingLineSearch(X, y, weight_W, weight_U,self.lamb, self.beta, PW, PU)

            del PW, PU

            new_weight_W = fc.fixOrthant(newLW,weight_W, new_weight_W)
            new_weight_U = fc.fixOrthant(newLU,weight_U, new_weight_U)
            loss = calLoss(X, y, new_weight_W, new_weight_U, self.lamb, self.beta)


                # 6. 判断是否提前终止
            if self.terminate and self.check(loss_before, loss_now):
                break
            else:
                overwrite = (it + 1) % self.memoryNum
                yList_u[overwrite] = LU - newLU
                yList_w[overwrite] = LW - newLW
                sList_u[overwrite] = new_weight_U - weight_U
                sList_w[overwrite] = new_weight_W - weight_W
                weight_U = new_weight_U
                weight_W = new_weight_W
                LW = newLW
                LU = newLU
                del newLW
                del newLU
                del new_weight_U
                del new_weight_W

            it += 1


        logging.info("loss: %s" % loss)
        logging.info("============iterator : %s end ==========" % it)
        print("")

        print("use time: ", time.time() - start_time)
        print("------------------------------------------------------\n")



        # with open("save/result"+self.stamp,"a") as fw:
        #     fw.write("train_acc:" + " ".join(ACC)+"\n")
        #     fw.write("train_loss:" + " ".join(LOSS) + "\n")
        #     fw.write("train_auc:" + " ".join(AUC) + "\n")
        #     fw.write("test_acc:" + " ".join(TEST_ACC) + "\n")
        #     fw.write("test_loss:" + " ".join(TEST_LOSS) + "\n")
        #     fw.write("test_auc:" + " ".join(TEST_AUC) + "\n")
        self.weight_W = weight_W
        self.weight_U = weight_U

    def predict_proba(self, X):
        return self._predict(X)

    def _mlr(self, x):
        return fc.mlr(self.weight_W,self.weight_U, (x, 1))

    def predict(self, X):
        return np.array(self._predict(X) > 0.5, dtype = int)



