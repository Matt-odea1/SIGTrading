#!/usr/bin/env python

import numpy as np
import pandas as pd
from GlobleTrading import getMyPosition as getPosition
from coint import get_matrices

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    print (df.shape)
    return (df.values).T

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

columns = [f"Day {i}" for i in range(nt)]
index = [f"Stock {i}" for i in range(nInst)]
df = pd.DataFrame(prcAll, index=index, columns=columns)
df_transposed = df.T

# Save the transposed DataFrame to a CSV file
output_csv = "./prices_exported.csv"
df_transposed.to_csv(output_csv)


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(250, 251): # (250, 501)
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        get_matrices(prcHistSoFar)


        # calc max amount allowed to hold ($10000 tops), then clip accordingly
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)

        # find change in volumes 1 day to the next
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)

        # sum total change in volume 
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume # tracking total amount traded

        comm = dvolume * commRate # commission value

        # updates cash 
        cash -= curPrices.dot(deltaPos) + comm
        print(cash)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)
