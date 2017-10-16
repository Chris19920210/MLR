from __future__ import division
import numpy as np
import time
import copy
import traceback

# with dense vector negative sample 0, postive sample 1
def performance(f):
    def fn(*args, **kw):
        t_start = time.time()
        r = f(*args, **kw)
        t_end = time.time()
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r
    return fn


def mlr_total(weight_W,weight_U,X):
    ux = np.dot(X, weight_U.T)
    e_x = np.exp(ux)
    e_x = e_x / e_x.sum(axis = 1)[0]
    sig = sigmoid(np.dot(X, weight_W.T))
    return (e_x * sig).sum(axis = 1)

# U is 2d-array with 2m*d dimension for each person item tuple of (x, label)

def mlr(W, U, x):
    """
    calculate mixture logistic regression
    :param U: m * d
    :param W: m * d
    :param x: d
    :return:
    """
    ux = np.dot(U, x)
    eux = softmax(ux)
    del ux
    return np.dot(eux, sigmoid(np.dot(W, x)))

def sigmoid(z):
    """
    calculate sigmoid
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))

def softmax(x):
    """
    softmax a array
    :param x:
    :return:
    """
    e_x = np.exp(x)
    return e_x / e_x.sum()

def calLoss(X, y, weight_W, weight_U, norm21, norm1):
    """
    :param data:
    :param weight_W:
    :param weight_U:
    :return:
    """
    functionLoss = calFunctionLoss(weight_W, weight_U, X, y)
    norm21Loss = calNorm21(weight_W + weight_U)
    norm1Loss = calNorm1(weight_W + weight_U)
    print(functionLoss , norm21 * norm21Loss , norm1 * norm1Loss)
    return functionLoss + norm21 * norm21Loss + norm1 * norm1Loss

def calFunctionLossOne(W_w, W_u,x, y):
    p = mlr(W_w, W_u, x)
    if y == 0:
        return - np.log(1 - p)
    else:
        return - np.log(p)


def calFunctionLoss(W_w, W_u, X, y):
    """
    calculate the loss over all data
    :param w_w:
    :param w_u:
    :param data:
    :return:
    """
    loss = map(lambda (x,y): calFunctionLossOne(W_w,W_u,x, y), zip(X, y))
    loss = sum(loss)
    return loss
    # print("loss is:  %s" % loss)

def calNorm21(weight):
    '''
    :param weight:
    :return:
    '''
    return (weight ** 2).sum() ** 0.5

def calNorm1(weight):
    """
    :param weight:
    :return:
    """
    return np.abs(weight).sum()

def calDimension21(W):
    """
    :param W:
    :return:{dimension1:std1, dimension2:std2 ......}
    """
    return (W**2).sum(axis = 0) ** 0.5

# derivative for each sample
def cal_derivative(W_w, W_u, x, y):
    """
    calculate derivative
    :param weight:
    :return:
    """
    ux = np.dot(W_u, x)
    eux = softmax(ux)
    del ux
    sig = sigmoid(np.dot(W_w, x))
    mlr = np.dot(eux, sig)
    prob_scalar =  - (y - mlr) / (mlr * (1 - mlr))
    dir_U = np.outer(prob_scalar * eux * (sig - mlr), x)
    dir_W = np.outer(prob_scalar * sig * (1 - sig) * eux, x)
    return dir_W, dir_U


def sumCalDerivative(WW, WU, X, y):
    all = map(lambda (x,y): cal_derivative(WW, WU,x, y), zip(X,y))
    LW, LU = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]),all,(0,0))
    return LW, LU




def virtualGradient(WW, WU, GW, GU,beta,lamb):
    """
    :param weight_W:
    :param weight_U:
    :param gradient_W:
    :param gradient_U:
    :param norm21:
    :param norm1:
    :return:
    """

    D21 = calDimension21(WW + WU)
    VW = calV(GW, beta)
    VU = calV(GU, beta)
    VD21 = calDimension21(VW + VU)
    sumVD21 = sum(VD21)
    DW = calDij(GW, WW, VW, D21, sumVD21, beta, lamb)
    DU = calDij(GU, WU, VU, D21, sumVD21, beta, lamb)
    return DW, DU


def calV(L, beta):
    """
    :param LW:
    :param LU:
    :param beta:
    :param lamb:
    :return:
    """
    V = np.copy(L)
    V = np.maximum(np.abs(V) - beta, 0)
    return V*np.sign(-L)

def calDij(L, W, V, D21, sumVD21, beta, lamb):
    mask1 = (W != 0)
    mask2 = (W == 0) * np.tile((D21 != 0), (len(W),1))
    mask3 = np.tile((D21 == 0), (len(W),1))
    D21_tmp = np.copy(D21)
    D21_tmp[D21_tmp == 0] = 1
    s = - L - lamb * W / D21_tmp
    cond1 =  s - beta * np.sign(W)
    cond2 = np.maximum(np.abs(s) - beta, 0.0)*np.sign(s)
    if (sumVD21 != 0 ):
        cond3 = V * (max(sumVD21 - lamb, 0.0) / sumVD21)
    else:
        cond3 = V * max(sumVD21 - lamb, 0.0)
    return mask1 * cond1 + mask2 * cond2 + mask3 * cond3


def loop(length, latest, direction):
    count = 0
    if(direction == 'right'):
        while(count < length):
            if(latest + count + 1 < length):
                yield latest + count + 1
                count += 1
            else:
                yield latest + count + 1 - length
                count += 1
    elif(direction == 'left'):
        while(count < length):
            if(latest - count >= 0):
                yield latest - count
                count += 1
            else:
                yield length + latest - count
                count += 1
    else:
        raise Exception("please enter left or right")


def adam(VW, VU,m_w, m_u, v_w, v_u, beta1, beta2, it, alpha, epison):
    m_w[::] = (beta1 * m_w - (1 - beta1) * VW)[::]
    m_u[::] = (beta1 * m_u - (1 - beta1) * VU)[::]
    v_w[::] = (beta2 * v_w + (1 - beta2) * (VW**2))[::]
    v_u[::] = (beta2 * v_u + (1 - beta2) * (VU**2))[::]
    m_w_hat = -m_w / (1 - beta1 ** it)
    m_u_hat = -m_u / (1 - beta1 ** it)
    mask_w = np.sign(m_w_hat) * np.sign(VW) > 0
    mask_u = np.sign(m_u_hat) * np.sign(VU) > 0
    m_w_hat = m_w_hat * mask_w
    m_u_hat = m_u_hat * mask_u
    v_w_hat = v_w / (1 - beta2**it)
    v_u_hat = v_u / (1 - beta2**it)
    return alpha * (m_w_hat / (v_w_hat ** 0.5 + epison)), alpha * (m_u_hat / (v_u_hat  ** 0.5 + epison))


## weight_w, weight_u, s
def lbfgs(VW, VU, sList_w,sList_u, yList_w, yList_u, k, m, start):
    """
    :param feaNum:
    :param gk : matrix, 2m*d
    :param sList:3d*matrix,steps * 2m * d
    :param yList:3d*matrix, steps * 2m * d
    :return:
    """
    if((sList_w[start] * yList_w[start] + sList_u[start] * yList_u[start]).sum() > 0 ):
        q_u = np.copy(VU)
        q_w = np.copy(VW)
        # for delta
        L = k + 1 if k < m else m
        alphaList = np.zeros(L)
        ro = (yList_w[(start - 1) % m] * sList_w[(start - 1) % m] + yList_u[(start - 1) % m] * sList_u[(start - 1) % m]).sum()
        print ("ro %f" % ro)

        for i in loop(L, start, 'left'):
            alpha = (sList_u[i] * q_u + sList_w[i] * q_w).sum() / (sList_u[i] * yList_u[i] + sList_w[i] * yList_u[i]).sum()
            print ("alpha %f" % alpha)
            if(alpha == np.nan or alpha == np.inf or alpha == -np.inf):
                return VW, VU
            q_u = q_u - alpha * yList_u[i]
            q_w = q_w - alpha * yList_w[i]
            alphaList[i] = alpha

        q_u = q_u * ro
        q_w = q_w * ro

        for i in loop(L,start,'right'):
            beta = (yList_u[i] * q_u + yList_w[i] * q_w).sum() / (sList_u[i] * yList_u[i] + sList_w[i] * yList_u[i]).sum()
            q_u = q_u + (alphaList[i] - beta) * sList_u[i]
            q_w = q_w + (alphaList[i] - beta) * sList_w[i]

        mask_u = np.sign(q_u) * np.sign(VU) > 0
        mask_w = np.sign(q_w) * np.sign(VW) > 0

        return q_w * mask_w, q_u * mask_u
    else:
        return VW, VU

def backTrackingLineSearch(X, y, weight_W, weight_U,norm21, norm1, pW, pU):
    """
    :param it:
    :param oldLoss:
    :param data:
    :param WW:
    :param WU:
    :param GW:
    :param GU:
    :param vGW:
    :param vGU:
    :return:
    """
    alpha = 1.0
    c = 0.5
    tao = 0.9
    LW, LU = sumCalDerivative(weight_W, weight_U, X, y)
    m = (pW * LW + pU * LU).sum()
    t = - c * m
    loss = calLoss(X, y, weight_W, weight_U, norm21, norm1)

    while True:
        newW = weight_W - alpha*pW
        newU = weight_U - alpha*pU

        new_loss = calLoss(X, y, newW, newU, norm21, norm1)

        if(loss > new_loss + alpha * t):
            return newW, newU
        else:
            alpha = tao * alpha

def fixOrthant(GW, weight_W, new_weight_W):
    mask = (weight_W == 0) * np.sign(GW) + (weight_W != 0) * np.sign(weight_W)
    mask = mask * new_weight_W > 0
    return new_weight_W * mask
