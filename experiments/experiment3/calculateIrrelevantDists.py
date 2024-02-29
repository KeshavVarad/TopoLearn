import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.integrate as integrate
import tqdm
import pandas as pd
import os

# Calculate average distance to irrelevant for every matrix and store it in 
    # data/experiment3/irrelevantDists/layer + str(layer_num) + "_head" + str(head_num) + ".csv"
    # with an id "label_name" + str(matrixNum)
    

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


def empirical_dist(x, set):

    return np.sum(set <= x) / len(set)

    

def birth_death_dist(z, adj):
    
    birth_set, death_set = get_birth_death_sets(adj)
    
    birth_val = 0
    death_val = 0
    for i in range(len(birth_set)):
        if i + 1 >= z * len(birth_set):
            birth_val = birth_set[i]
            break
        
    for i in range(len(death_set)):
        if i + 1 >= z * len(death_set):
            death_val = death_set[i]
            break
    
    return birth_val, death_val


def birth_square_diff(z, adj1, adj2):
    
    birth_dist1, _ = birth_death_dist(z, adj1)
    birth_dist2, _ = birth_death_dist(z, adj2)
    
    square_diff = (birth_dist2 - birth_dist1) ** 2
    
    return square_diff

def death_square_diff(z, adj1, adj2):
    
    _, death_dist1 = birth_death_dist(z, adj1)
    _, death_dist2 = birth_death_dist(z, adj2)
    
    square_diff = (death_dist1 - death_dist2) ** 2
    
    return square_diff

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
    
    
    
# def top_loss_0(adj1, adj2):
#     res, error = integrate.quad(lambda x: birth_square_diff(x, adj1, adj2), 0, 1)
#     return res

# def top_loss_1(adj1, adj2):
#     res, error = integrate.quad(lambda x: death_square_diff(x, adj1, adj2), 0, 1)
#     return res


    
PATH_TO_DATA = "/Users/keshav/Desktop/Github/TopoLearn/data/"
SAVE_AFTER = 100

for layer_num in tqdm.tqdm(range(12), "Calculating distances"):
    PATH_TO_MATRIX = PATH_TO_DATA + "averageAttentionMatrices/layer" + str(layer_num) + "/"
    PATH_TO_IRRELEVANT = PATH_TO_MATRIX + "irrelevant/"
    PATH_TO_STEREOTYPE = PATH_TO_MATRIX + "stereotype/"
    PATH_TO_ANTISTEREOTYPE = PATH_TO_MATRIX + "antistereotype/"
    
    irrelevantMatrixFiles = os.listdir(PATH_TO_IRRELEVANT)
    stereotypeMatrixFiles = os.listdir(PATH_TO_STEREOTYPE)
    antistereotypeMatrixFiles = os.listdir(PATH_TO_ANTISTEREOTYPE)
    
    stereotypeMatrix = None
    antistereotypeMatrix = None
    
    
    distList = []
    
    for matrix_num in tqdm.tqdm(range(len(irrelevantMatrixFiles)), desc = "Processing layer " + str(layer_num), leave = False):
        
        stereotypeFileName = PATH_TO_STEREOTYPE + stereotypeMatrixFiles[matrix_num]
        antistereotypeFileName = PATH_TO_ANTISTEREOTYPE + antistereotypeMatrixFiles[matrix_num]
        
        
        
        with open(stereotypeFileName, "rb") as f:
            stereotypeMatrix = np.load(f)
            stereotypeMatrix = stereotypeMatrix.astype(float)
        with open(antistereotypeFileName, "rb") as f:
            antistereotypeMatrix = np.load(f)
            antistereotypeMatrix = antistereotypeMatrix.astype(float)
        
    
        
        averageStereotypeDist0 = 0
        averageStereotypeDist1 = 0
        averageAntistereotypeDist0 = 0
        averageAntistereotypeDist1 = 0
        
        
        for irrelevantMatrixNum in tqdm.tqdm(range(len(irrelevantMatrixFiles)), desc = "Calculating distances to irrelevant", leave = False):
            irrelevantFileName = PATH_TO_IRRELEVANT + irrelevantMatrixFiles[irrelevantMatrixNum]
            
            with open(irrelevantFileName, "rb") as f:
                irrelevantMatrix = np.load(f)
                irrelevantMatrix = irrelevantMatrix.astype(float)
            
            stereotypeDist0, stereotypeDist1 = top_loss(stereotypeMatrix, irrelevantMatrix)
            antistereotypeDist0, antistereotypeDist1 = top_loss(antistereotypeMatrix, irrelevantMatrix)
            
            averageStereotypeDist0 += stereotypeDist0
            averageStereotypeDist1 += stereotypeDist1
            
            averageAntistereotypeDist0 += antistereotypeDist0
            averageAntistereotypeDist1 += antistereotypeDist1
        
        averageStereotypeDist0 /= len(irrelevantMatrixFiles)
        averageStereotypeDist1 /= len(irrelevantMatrixFiles)
        averageAntistereotypeDist0 /= len(irrelevantMatrixFiles)
        averageAntistereotypeDist1 /= len(irrelevantMatrixFiles)
        
        stereotypeDistDict = {
            "Matrix ID": "stereotype" + str(matrix_num),
            "0D Dist": averageStereotypeDist0,
            "1D Dist": averageStereotypeDist1,
        }
        
        antistereotypeDistDict = {
            "Matrix ID": "antistereotype" + str(matrix_num),
            "0D Dist": averageAntistereotypeDist0,
            "1D Dist": averageAntistereotypeDist1,
        }
        
        distList.append(stereotypeDistDict)
        distList.append(antistereotypeDistDict)
    
        
        if ((matrix_num + 1) % SAVE_AFTER) == 0:
            distDf = pd.DataFrame(distList)
            
            distDf.to_csv(PATH_TO_DATA + "experiment3/irrelevantDists/layer" + str(layer_num) + ".csv", index=False)
            
