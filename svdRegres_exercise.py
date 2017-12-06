import numpy as np


def ecludSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    return 0.5 * (1 + np.corrcoef(inA, inB, rowvar=0)[0][1])


def cosSim(inA, inB):
    num = float(np.dot(inA.T, inB))
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 * (1 + num / denom)


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def standEst(dataMat, user, simMeas, item):
    m, n = dataMat.shape
    # n = food dishes
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue

        # j = 5 , 10

        overLap = np.nonzero(np.logical_and(dataMat[:, item] > 0, dataMat[:, j] > 0))[0]
        
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, eatMethod=standEst):
    unratedItems = np.nonzero(dataMat[user] == 0)[0]
    if len(unratedItems) == 0:
        return 'u rated everyting!'

    itemScores = []
    for item in unratedItems:
        estimatedScore = standEst(dataMat, user, cosSim, item)
        itemScores.append((item, estimatedScore))
            
    sorted_itemScores = sorted(itemScores, key=lambda jj: jj[1], reverse=True)
    return sorted_itemScores[:N]


def use_svd(dataMat):
    U, sigma, VT = np.linalg.svd(dataMat)
    
    Sig2 = sigma ** 2
    sum_Sig2 = sum(Sig2) * 0.9
    print(sum(Sig2[:3]) > sum_Sig2)
    print(sum(Sig2[:3]))


def svdEst(dataMat, user, simMeas, item):
    m, n = dataMat.shape
    # n = food dishes
    simTotal = 0.0
    ratSimTotal = 0.0

    U, sigma, VT = np.linalg.svd(dataMat)
    sig4 = np.eye(4) * sigma[:4]

    xformedItems = dataMat.T * np.mat(U[:, :4]) * sig4

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue

        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)

        simTotal += similarity
        ratSimTotal += similarity * userRating

    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal



if __name__ == '__main__':
    dataMat = loadExData2()
    dataMat_1 = loadExData()
    dataMat = np.array(dataMat)
    dataMat_1 = np.array(dataMat_1)

    U, sigma, VT = np.linalg.svd(dataMat_1)
    # print(U)
    # print(sigma)
    # print(VT)

    user = 0

    unratedItems = np.nonzero(dataMat[user] == 0)[0]

    recommend_result = recommend(dataMat, user)

    # print(recommend_result)
    # use_svd_recommend_result = [(1, 4.5269613419755483), (2, 4.5269440375646592), (0, 4.5269187296713289)]
    # common_recommend_result = [(0, 5.0), (1, 5.0), (2, 5.0)]

    pass