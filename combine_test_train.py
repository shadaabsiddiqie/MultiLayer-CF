import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
import csv
from math import pi ,sin ,cos ,atan2,sqrt,exp
from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()

trainRatingInfo = ['user_id', 'item_id', 'rating', 'timestamp']
dfTrainRating = pd.read_csv('u4.base',sep='\t',names=trainRatingInfo)

testRatingInfo = ['user_id', 'item_id', 'rating', 'timestamp']
dfTestRating = pd.read_csv('u4.test',sep='\t',names=testRatingInfo)

ItemInfo = ["movie_id","movie_title","release_date","video_release_date","IMDb URL","unknown","Action",
            "Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy",
            "FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western"]
dfItem = pd.read_csv('u.item',sep='\|',engine='python',names=ItemInfo)

UserInfo =  ["user_id","age","gender","occupation","zip_code"]
dfUser = pd.read_csv('u.user',sep='\|',engine='python',names=UserInfo)

ItemDict = {}
for row in dfItem.itertuples():
    ItemDict[row[1]] = row

UserDict = {}
for row in dfUser.itertuples():
    UserDict[row[1]] = row

def haversine(pos1, pos2):
    lat1 = pos1[0]
    long1 =pos1[1]
    lat2 = pos2[0]
    long2 = pos2[1]
    degree_to_rad = float(pi / 180.0)
    d_lat = (lat2 - lat1) * degree_to_rad
    d_long = (long2 - long1) * degree_to_rad
    a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    km = 6367 * c                                   # mi = 3956 * c
    return km                                       # return {"km":km, "miles":mi}

def similarityDist(lalo):
    n_users = dfUser.user_id.unique().shape[0]
    distAB = np.zeros((n_users, n_users))
    for i in range(1,n_users+1):
        for j in range(1,n_users+1):
            dist = haversine(lalo[i], lalo[j])
            if dist <= 1000:
                distAB[i-1][j-1] = 1
    return (distAB)

def similarityAge():
    n_users = dfUser.user_id.unique().shape[0]
    AgeAB = np.zeros((n_users, n_users))
    OccupationAB = np.zeros((n_users, n_users))
    GenderAB = np.zeros((n_users, n_users))
    for i in range(1,n_users+1):
        for j in range(1,n_users+1):
            if UserDict[i].occupation == UserDict[j].occupation and UserDict[i].occupation!="other" and UserDict[i].occupation!="none":
                OccupationAB[i-1][j-1] = 1
            if UserDict[i].gender == UserDict[j].gender:
                GenderAB[i-1][j-1] = 1
            if UserDict[i].age in range(0,39) and UserDict[j].age in range(0,39):
                AgeAB[i-1][j-1] = 1
            elif UserDict[i].age in range(40,64) and UserDict[j].age in range(40,64):
                AgeAB[i-1][j-1] = 1
            elif UserDict[i].age in range(65,150) and UserDict[j].age in range(65,150):
                AgeAB[i-1][j-1] = 1
    return AgeAB , OccupationAB ,GenderAB

def matrixToOnes(Mat):
    oneMat = np.zeros((943, 1682))
    for x in range(Mat.shape[0]):
        row = Mat[x]
        nz = row[row>0]
        nzSum = nz.sum()
        nzSize = nz.size
        R = np.zeros((1682))
        if nzSize == 0:
            oneMat[x] = R
        else:
            nzAvg = int(nzSum/nzSize)
            for c in range(1682):
                if row[c]>=nzAvg :
                    R[c] = 1
                elif row[c]<nzAvg and row[c]>0:
                    R[c] = -1
            oneMat[x] = R
    return oneMat

def precision(noOfHits, noOfRecos):
    return noOfHits/noOfRecos

def recall(noOfHits, noOfTestCases):
    return noOfHits/noOfTestCases

def f1Metric(precision,recall):
    if precision == 0 or recall == 0:
        return 0
    return ((2*precision*recall)/(precision+recall))

def evaluate(recommendationsList,onetestX, noOfRecos=10):
    testRatingsOfUser = np.where(onetestX>0)[0]
    O = onetestX[onetestX==1]
    if O.shape[0] == 0:
        return (0,0,0,0)
    N = onetestX[onetestX==-1]
    testSize = O.shape[0]+N.shape[0]
    noOfHits = 0
    for recommendation in recommendationsList[:noOfRecos]:
        if recommendation in testRatingsOfUser and onetestX[recommendation]>0:
            noOfHits+=1
    p = precision(noOfHits, noOfRecos)
    r = recall(noOfHits,testSize)
    f1 = f1Metric(p, r)
    return (p,r,f1,noOfHits)

n_users      = 943                                #no of users
n_items      = 1682                               #no of items
train        = np.zeros((943, 1682))              #Matrix of users,items with there rating
test         = np.zeros((943, 1682))              #Matrix of users,items with there rating
onetrain     = np.zeros((943, 1682))              #Matrix of users,items with there rating with one's
onetest      = np.zeros((943, 1682))              #Matrix of users,items with there rating with one's
genreSum     = np.zeros((n_users, 18))            #Matrix of users,genre with there sum of all rating for each genre
genreFreq    = np.zeros((n_users, 18))            #Matrix of users,genre with there frequency of genre
genre        = np.zeros((n_users, 18))            #Matrix of users,genre with there avg.genre values
           
for row in dfTrainRating.itertuples():
    train[row[1]-1, row[2]-1] = row[3]
for row in dfTestRating.itertuples():
    test[row[1]-1, row[2]-1] = row[3]

onetrain = matrixToOnes(train)
onetest = matrixToOnes(test)

# MDMatrix = np.genfromtxt('MovieDirector.csv', delimiter=',') #MDMatrix     = np.zeros((1682, 1245))  
# Directors = np.dot(train,MDMatrix)                           #Directors    = np.zeros((943, 1245))

MCMatrix = np.genfromtxt('MovieCast.csv', delimiter=',')  #MCMatrix     = np.zeros((1682, 54853))
CastMembers = np.dot(train,MCMatrix)                      #CastMembers  = np.zeros((943,54853))

# MC10Matrix = np.genfromtxt('MovieCast10.csv', delimiter=',') 
# CastMembers10 = np.dot(train,MC10Matrix)

# MC15Matrix = np.genfromtxt('MovieCast15.csv', delimiter=',') 
# CastMembers15 = np.dot(train,MC15Matrix)

# print(Directors.shape)
# print(Directors[1][:10])
# uId = 0
# for row in onetrain:
#     for x in range(row.shape[0]):
#         if row[x]==1:
#             itemDetails = ItemDict[x+1]
#             for i in range(0,18):
#                 if(itemDetails[i+7] == 1):
#                     genre[uId,i]=genre[uId,i]+train[uId,x]
#     uId = uId + 1        
    
# for row in dfTrainRating.itertuples():
#     itemDetails = ItemDict[row[2]]
#     for x in range(0,18):
#         if(itemDetails[x+7] == 1):
#             if row[3] >= 2:
#                 genreSum[row[1]-1,x]=row[3] +genreSum[row[1]-1,x]
#                 genreFreq[row[1]-1,x] =genreFreq[row[1]-1,x]+1
#             if genreFreq[row[1]-1,x] == 0:
#                 genre[row[1]-1,x] = 0
#             else:
#                 genre[row[1]-1,x]= genreSum[row[1]-1,x]/genreFreq[row[1]-1,x]
# print(genre)

def fast_similarity(ratings, epsilon=1e-9):
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

i = 1
lalo = {}
a = []
for row in dfUser.itertuples():
    la = 0
    lo = 0
    try:
        lo = zcdb[UserDict[i].zip_code].longitude
        la = zcdb[UserDict[i].zip_code].latitude
    except:
        a.append(UserDict[i].zip_code)
    lalo[i] = (la,lo)
    i = i + 1

R_similarity = fast_similarity(train)
G_similarity = fast_similarity(genre)
Gf_similarity = fast_similarity(genreFreq)
Gs_similarity = fast_similarity(genreSum)
L_similarity = similarityDist(lalo)
A_similarity , O_similarity ,S_similarity= similarityAge()
# D_similarity = fast_similarity(Directors)
C_similarity = fast_similarity(CastMembers)
# C10_similarity = fast_similarity(CastMembers10)
# C15_similarity = fast_similarity(CastMembers15)

def ChangedPred(pred , train):
    pred2 = np.zeros(pred.shape)
    for x in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            if train[x][y] == 0:
                pred2[x][y] = pred[x][y]
    return pred2

def predict_topk(ratings, similarity, k=100):
    pred = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            x = similarity[i, :][top_k_users]
            y = ratings[:, j][top_k_users]
            pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
            pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    return pred

def predict_fast_simple(ratings, similarity):
    return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T

def Intercetion(pred1 , pred2 ,n):
    iS = 0
    cm = 0
    for x in range(pred1.shape[0]):
        l1 = pred1[x].argsort()[-n:][::-1]
        l2 = pred2[x].argsort()[-n:][::-1]
        c = 0
        for i in l2:
            if i in l1:
                iS = iS + 1
                cm = cm + c
            c = c + 1
    return (iS/pred1.shape[0],cm/pred1.shape[0]) 

def combRecommendation(recLa,recLb,beta,n):
    a = int(n*beta)
    b = int(n*(1-beta))
    rLa = recLa[:a]
    # rLb = recLb[:b]
    rLb = np.ones(b)
    added = 0
    index = 0
    while(True):
        if recLb[index] not in rLa:
            rLb[added] = recLb[index]
            added = added + 1
        index = index + 1 
        if added == b :
            break
    ans = np.concatenate([rLa,rLb])
    return ans.astype(int)

a = 10 ; b = 5 ; c = 9 ; d = 0 ; e = 0

O = O_similarity
A = A_similarity
L = L_similarity
S = S_similarity
comb = O+A+L+S
# comb = comb.astype(int)

#Gs 6 ->0.184
# 10 3 2
#Gf 1 1 ->0.186
#only comb2 ->0.0922
#only genre -> 0.126

P = np.zeros((943))
R = np.zeros((943))
F1= np.zeros((943))
Hits= np.zeros((943))

tP = np.zeros((15))
tR = np.zeros((15))
tF1 = np.zeros((15))

for b in range(15):
# for c in range(11):
    Similarity = R_similarity+C_similarity
    # PredictionR = predict_topk(train, R_similarity,k=(b+1)*10)
    # PredictionC = predict_topk(train, C_similarity,k=(b+1)*10)
    # PredictionR = ChangedPred(PredictionR,train)
    # PredictionC = ChangedPred(PredictionC,train)
    # print("Intersect = ",Intercetion(PredictionR , PredictionC ,10))
    Prediction = predict_topk(train, Similarity,k=(b+1)*10)
    Prediction = ChangedPred(Prediction,train)
    
    for x in range(943):
        # recommendationsListR = np.argsort(PredictionR[x])[::-1]
        # recommendationsListC = np.argsort(PredictionC[x])[::-1]
        # recommendationsList = combRecommendation(recommendationsListR,recommendationsListC,1,10)
        recommendationsList = np.argsort(Prediction[x])[::-1]
        P[x],R[x],F1[x],Hits[x] = evaluate(recommendationsList,onetest[x], noOfRecos=10)
    
    Precition = sum(P)/len(P)
    Recall = sum(R)/len(R)
    F1_Score = sum(F1)/len(F1)

    tP[b]=Precition
    tR[b]=Recall
    tF1[b]=F1_Score

    print("K= c= ",(b+1)*10,c/10)
    print("p= ",Precition)
    print("r= ",Recall)
    print("f1= ",F1_Score)
    print("----------------------------------")

print("Precition")
for x in tP:
    print(x)

print("Recall")
for x in tR:
    print(x)

print("F1_Score")
for x in tF1:
    print(x)
