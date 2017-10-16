import numpy as np
import copy
import pickle
import time
import random
import logging
from function import *

class LSPLM:

    def __init__(self,
                 pieceNum = 12,
                 iterNum = 1000,
                 intercept = True,
                 beta1 = 0.1,
                 beta2 = 0.1,
                 alpha = 0.001,
                 epison = 10e-8,
                 lamb = 0.1,
                 beta = 0.1,
                 terminate = False
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
        self.beta = beta
        self.lamb = lamb
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.N = 0
        self.p = 0
        self.terminate = terminate
        self.epison = epison

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

        ## Intialization
        np.random.seed(0)
        weight_W = np.random.normal(0,0.1, (self.pieceNum, self.p))
        weight_U = np.random.normal(0,0.1, (self.pieceNum, self.p))
        best_weight_W = np.random.normal(0,0.1, (self.pieceNum, self.p))
        best_weight_U = np.random.normal(0,0.1, (self.pieceNum, self.p))
        m_w = np.zeros((self.pieceNum, self.p))
        m_u = np.zeros((self.pieceNum, self.p))
        v_w = np.zeros((self.pieceNum, self.p))
        v_u = np.zeros((self.pieceNum, self.p))
        loss_before = calLoss(X, y, weight_W, weight_U, self.lamb, self.beta)
        GW, GU = sumCalDerivative(weight_W, weight_U, X, y)
        LW, LU = virtualGradient(weight_W, weight_U, GW, GU,self.beta,self.lamb)
        it = 1
        del GW,GU
        loss_best = np.maximum
        best_iter = 0
        optimize_counter = 0
        while it < self.iterNum:
            print "iter:%d, loss:%s" % (it, loss_before)
            start_time = time.time()

            GW, GU = sumCalDerivative(weight_W, weight_U, X, y)
            newLW, newLU = virtualGradient(weight_W, weight_U, GW, GU,self.beta,self.lamb)
            del GW,GU


            m_w, m_u, v_w, v_u, PW, PU = adam(newLW, newLU, m_w, m_u, v_w, v_u, self.beta1, self.beta2, it, self.alpha, self.epison)

            new_weight_W, new_weight_U = weight_W + PW, weight_U + PU

            del PW, PU

            new_weight_W = fixOrthant(newLW, weight_W, new_weight_W)
            new_weight_U = fixOrthant(newLU, weight_U, new_weight_U)
            loss_now = calLoss(X, y, new_weight_W, new_weight_U, self.lamb, self.beta)
            if(loss_now < loss_best):
                loss_best = loss_now
                best_iter = it
                best_weight_W = new_weight_W
                best_weight_U = new_weight_U
            if(loss_before < loss_now):
                optimize_counter += 1
                if(optimize_counter >= 10):
                    self.weight_U = best_weight_U
                    self.weight_W = best_weight_W
                    print("use time: ", time.time() - start_time)
                    print("The best result get at %d iteration with loss: %f" % (best_iter, loss_best))
                    return "Done!"
            else:
                optimize_counter = 0


            weight_U = new_weight_U
            weight_W = new_weight_W
            LW = newLW
            LU = newLU
            del newLW
            del newLU
            del new_weight_U
            del new_weight_W
            loss_before = loss_now


            it += 1


        logging.info("============iterator : %s end ==========" % it)
        print("")

        print("The best result get at %d iteration with loss: %f" % (best_iter, loss_best))
        print("Done!")
        self.weight_W = best_weight_W
        self.weight_U = best_weight_U

    def predict_proba(self, X):
        N, p = X.shape
        if self.intercept:
            pad = np.ones((N, p + 1))
            pad[:,:-1] = X
            X = pad
            del pad
        return np.array(map(lambda x: mlr(self.weight_W,self.weight_U,x),X))


    def predict(self, X):
        return np.array(self.predict_proba(X) > 0.5, dtype = int)
