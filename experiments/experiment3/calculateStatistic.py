import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import tqdm
import pandas as pd
import os


PATH_TO_DATA = "/Users/keshav/Desktop/Github/TopoLearn/data/"


def birth_death_decomp(adj):
    eps = np.nextafter(0, 1)
    adj[adj == 0] = eps
    adj = np.triu(np.transpose(adj), k=1)

    Xcsr = csr_matrix(-adj)
    Tcsr = minimum_spanning_tree(Xcsr)
    mst = -Tcsr.toarray()  # reverse the negative sign
    nonmst = adj - mst
    
    return mst, nonmst

def get_birth_death_sets(adj):
    mst, nonmst = birth_death_decomp(adj)
    
    
    birth_ind = np.nonzero(mst)
    death_ind = np.nonzero(nonmst)
    return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])


def integrateSquareDiff(set1, set2):
        winSize = 1 / np.gcd(len(set1), len(set2))
        
        res = 0
        
        cur = winSize
        ind1 = 0
        ind2 = 0
        
        nextStep1 = 1 / len(set1)
        nextStep2 = 1 / len(set2)
        
        while cur <= 1:
            
            if cur > nextStep1:
                ind1 += 1
                nextStep1 += 1 / len(set1)
            
            if cur > nextStep2:
                ind2 += 1
                nextStep2 += 1 / len(set2)
            
            squareDiff = (set1[ind1] - set2[ind2]) ** 2
            res += squareDiff * winSize
            cur += winSize

        return res
        
    

def top_loss(adj1, adj2):
    bs1, ds1 = get_birth_death_sets(adj1)
    bs2, ds2 = get_birth_death_sets(adj2)
    
    
    loss0 = integrateSquareDiff(bs1, bs2)
    loss1 = integrateSquareDiff(ds1, ds2)
    
    return loss0, loss1


def initializeMemory(layerNum):
    
    
    PATH_TO_STEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layerNum) + "/stereotype/"
    PATH_TO_ANTISTEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layerNum) + "/antistereotype/"
    
    stereotypeMatrixFiles = os.listdir(PATH_TO_STEREOTYPE)
    antistereotypeMatrixFiles = os.listdir(PATH_TO_ANTISTEREOTYPE)
    
    SIZE = len(stereotypeMatrixFiles)
    
    memStereotype0 = np.zeros(shape = (SIZE, SIZE))
    memAntistereotype0 = np.zeros(shape = (SIZE, SIZE))
    
    memStereotype1 = np.zeros(shape = (SIZE, SIZE))
    memAntistereotype1 = np.zeros(shape = (SIZE, SIZE))
    
    stereotypeMatrix1 = None
    stereotypeMatrix2 = None
    antistereotypeMatrix1 = None
    antistereotypeMatrix2 = None
    
    
    for matrixNum1 in tqdm.tqdm(range(SIZE), desc = "Calculating memory matrices"):
        for matrixNum2 in tqdm.tqdm(range(matrixNum1 + 1, SIZE), desc = "Processing data point " + str(matrixNum1)):
            
            stereotypeFile1 = PATH_TO_STEREOTYPE + stereotypeMatrixFiles[matrixNum1]
            stereotypeFile2 = PATH_TO_STEREOTYPE + stereotypeMatrixFiles[matrixNum2]
            antistereotypeFile1 = PATH_TO_ANTISTEREOTYPE + antistereotypeMatrixFiles[matrixNum1]
            antistereotypeFile2 = PATH_TO_ANTISTEREOTYPE + antistereotypeMatrixFiles[matrixNum2]
            
            
            with open(stereotypeFile1, "rb") as f:
                stereotypeMatrix1 = np.load(f)
            with open(stereotypeFile2, "rb") as f:
                stereotypeMatrix2 = np.load(f)
                
            with open(antistereotypeFile1, "rb") as f:
                antistereotypeMatrix1 = np.load(f)
            with open(antistereotypeFile2, "rb") as f:
                antistereotypeMatrix2 = np.load(f)
            
            stereotypeLoss0, stereotypeLoss1 = top_loss(stereotypeMatrix1, stereotypeMatrix2)
            antistereotypeLoss0, antistereotypeLoss1 = top_loss(antistereotypeMatrix1, antistereotypeMatrix2)
            
            memStereotype0[matrixNum1][matrixNum2] = stereotypeLoss0
            memStereotype0[matrixNum2][matrixNum1] = stereotypeLoss0
            
            memAntistereotype0[matrixNum1][matrixNum2] = antistereotypeLoss0
            memAntistereotype0[matrixNum2][matrixNum1] = antistereotypeLoss0
            
            memStereotype1[matrixNum1][matrixNum2] = stereotypeLoss1
            memStereotype1[matrixNum2][matrixNum1] = stereotypeLoss1
            
            memAntistereotype1[matrixNum1][matrixNum2] = antistereotypeLoss1
            memAntistereotype1[matrixNum2][matrixNum1] = antistereotypeLoss1
    
    
    return memStereotype0, memStereotype1, memAntistereotype0, memAntistereotype1

def updateMemory(updateIndices, layerNum, cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1):
    PATH_TO_STEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layerNum) + "/stereotype/"
    PATH_TO_ANTISTEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layerNum) + "/antistereotype/"
    
    stereotypeMatrixFiles = os.listdir(PATH_TO_STEREOTYPE)
    antistereotypeMatrixFiles = os.listdir(PATH_TO_ANTISTEREOTYPE)
    
    SIZE = len(stereotypeMatrixFiles)
    
    stereotypeMatrix1 = None
    stereotypeMatrix2 = None
    antistereotypeMatrix1 = None
    antistereotypeMatrix2 = None
    
    for matrixNum1 in updateIndices:
        for matrixNum2 in range(SIZE):
            if matrixNum1 == matrixNum2:
                continue
            
            stereotypeFile1 = PATH_TO_STEREOTYPE + stereotypeMatrixFiles[matrixNum1]
            stereotypeFile2 = PATH_TO_STEREOTYPE + stereotypeMatrixFiles[matrixNum2]
            antistereotypeFile1 = PATH_TO_ANTISTEREOTYPE + antistereotypeMatrixFiles[matrixNum1]
            antistereotypeFile2 = PATH_TO_ANTISTEREOTYPE + antistereotypeMatrixFiles[matrixNum2]
            
            
            with open(stereotypeFile1, "rb") as f:
                stereotypeMatrix1 = np.load(f)
            with open(stereotypeFile2, "rb") as f:
                stereotypeMatrix2 = np.load(f)
                
            with open(antistereotypeFile1, "rb") as f:
                antistereotypeMatrix1 = np.load(f)
            with open(antistereotypeFile2, "rb") as f:
                antistereotypeMatrix2 = np.load(f)
            
            stereotypeLoss0, stereotypeLoss1 = top_loss(stereotypeMatrix1, stereotypeMatrix2)
            antistereotypeLoss0, antistereotypeLoss1 = top_loss(antistereotypeMatrix1, antistereotypeMatrix2)
            
            cluster1Memory0[matrixNum1][matrixNum2] = stereotypeLoss0
            cluster1Memory0[matrixNum2][matrixNum1] = stereotypeLoss0
            
            cluster2Memory0[matrixNum1][matrixNum2] = antistereotypeLoss0
            cluster2Memory0[matrixNum2][matrixNum1] = antistereotypeLoss0
            
            cluster1Memory1[matrixNum1][matrixNum2] = stereotypeLoss1
            cluster1Memory1[matrixNum2][matrixNum1] = stereotypeLoss1
            
            cluster2Memory1[matrixNum1][matrixNum2] = antistereotypeLoss1
            cluster2Memory1[matrixNum2][matrixNum1] = antistereotypeLoss1
    
    return cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1
            

def calculateStatistic(layerNum, cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1, cluster1Files, cluster2Files):
    
    irrelevantDists = pd.read_csv(PATH_TO_DATA + "experiment3/irrelevantDists/layer" + str(layerNum) + ".csv", index_col=0)
    irrelevantDists.set
    
    average0DistToCluster1 = np.mean(cluster1Memory0, axis = 1)
    average0DistToCluster2 = np.mean(cluster2Memory0, axis = 1)
    
    average1DistToCluster1 = np.mean(cluster1Memory1, axis = 1)
    average1DistToCluster2 = np.mean(cluster2Memory1, axis = 1)
    
    cluster1Class = ""
    cluster2Class = ""
    
    delta0Cluster1 = np.zeros_like(average0DistToCluster1)
    delta0Cluster2 = np.zeros_like(average0DistToCluster2)
    delta1Cluster1 = np.zeros_like(average1DistToCluster1)
    delta1Cluster2 = np.zeros_like(average1DistToCluster2)
    
    
    for matrixNum in tqdm.tqdm(range(len(average0DistToCluster1)), desc = "Calculating statistic", leave = False):
        
        if "stereotype" in cluster1Files[matrixNum]:
            cluster1Class = "stereotype"
        else:
            cluster1Class = "antistereotype"
        
        if "stereotype" in cluster2Files[matrixNum]:
            cluster2Class = "stereotype"
        else:
            cluster2Class = "antistereotype"

        cluster1Id = cluster1Class + str(matrixNum)
        cluster2Id = cluster2Class + str(matrixNum)
        
        cluster1IrrelevantDist0 = irrelevantDists[cluster1Id]["0D Dist"]
        cluster1IrrelevantDist1 = irrelevantDists[cluster1Id]["1D Dist"]
        cluster2IrrelevantDist0 = irrelevantDists[cluster2Id]["0D Dist"]
        cluster2IrrelevantDist1 = irrelevantDists[cluster2Id]["1D Dist"]
        
        delta0Cluster1[matrixNum] = average0DistToCluster1[matrixNum] / cluster1IrrelevantDist0
        delta0Cluster2[matrixNum] = average0DistToCluster2[matrixNum] / cluster2IrrelevantDist0
        delta1Cluster1[matrixNum] = average1DistToCluster1[matrixNum] / cluster1IrrelevantDist1
        delta1Cluster2[matrixNum] = average1DistToCluster2[matrixNum] / cluster2IrrelevantDist1
    
    cluster1Statistic0 = np.mean(delta0Cluster1)
    cluster2Statistic0 = np.mean(delta0Cluster2)
    
    cluster1Statistic1 = np.mean(delta1Cluster1)
    cluster2Statistic1 = np.mean(delta1Cluster2)
    
    stat0 = cluster2Statistic0 - cluster1Statistic0
    stat1 = cluster2Statistic1 - cluster1Statistic1
    
    return stat0, stat1        


NUM_ITERATIONS = 1000
PERMUTE_RATIO = 0.1
SAVE_AFTER = 100

for layer_num in range(12):
    
    layer_statistics0 = []
    layer_statistics1 = []

    PATH_TO_STEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layer_num) + "/stereotype/"
    PATH_TO_ANTISTEREOTYPE = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layer_num) + "/antistereotype/"

    cluster1Files = os.listdir(PATH_TO_STEREOTYPE)
    cluster2Files = os.listdir(PATH_TO_ANTISTEREOTYPE)
    
    SIZE = len(cluster1Files)
    
    cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1 = initializeMemory(layer_num)
    
    for iter in range(NUM_ITERATIONS):
        stat0, stat1 = calculateStatistic(layer_num, cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1, cluster1Files, cluster2Files)
        layer_statistics0.append(stat0)
        layer_statistics1.append(stat1)
        
        if (iter + 1 % SAVE_AFTER) == 0:
            save_dir = PATH_TO_DATA + "experiment3/statistics/layer" + str(layer_num)
            stat0Array = np.array(stat0)
            stat1Array = np.array(stat1)
            
            stackedStats = np.stack([stat0, stat1], axis = 0)
            
            with open(save_dir, "wb") as f:
                np.save(f, stackedStats)
            
        
        permute_indices = np.random.choice(SIZE, size = int(SIZE * PERMUTE_RATIO), replace=False)
        
        for permute_indice in permute_indices:
            temp = cluster1Files[permute_indice]
            cluster1Files[permute_indice] = cluster2Files[permute_indice]
            cluster2Files[permute_indice] = temp
        
        cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1 = updateMemory(permute_indices, layer_num, cluster1Memory0, cluster1Memory1, cluster2Memory0, cluster2Memory1)

