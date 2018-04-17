#!/usr/bin/env python
import numpy as np
from math import log
from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    c = 0
    h = sig(np.matmul(x, theta))
    for hi, yi in zip(h, y) :
        if yi == 0:
            c += -1 * log(1-hi)
        else:
            c += -1 * log(hi)
    c = c/N
    return np.array([c])


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #

    g = np.zeros(theta.shape)

    # END TODO
    ###########

    return g
