#!/usr/bin/env python
# coding: utf-8

# # AI - Project
# Muhammad Fateh Tariq - FA17_BSCS-019 | 
# Asad Ali Hasani - FA17_BSCS-020 | 
# Osama Syed - FA17_BSCS-069 | 
# Salman Ahmed - FA17_BSCS-082

# In[57]:


import math as mt
import pandas as pd
import numpy as np
import copy as copy
import sklearn.metrics


# In[83]:


StudentInfo = pd.read_excel('Book1.xlsx')
print('Student Info: ')

print(StudentInfo)


arr = StudentInfo.to_numpy()
arrT = np.transpose(arr)

print('\n' + str(arr))


# # Similarity Functions

# In[59]:


def cosineDeff(v1, v2):
    aIntob = 0
    aSquare = 0
    bSquare = 0

    if(len(v1) == len(v2)):
        for i in range(len(v1)):
            aIntob += v1[i] * v2[i]
            aSquare += v1[i] * v1[i]
            bSquare += v2[i] * v2[i]

    return aIntob / (mt.sqrt(aSquare) * mt.sqrt(bSquare))

def AvgCossimDeff(v1, v2):
    aIntob = 0
    aSquare = 0
    bSquare = 0

    if(len(v1) == len(v2)):
        for i in range(len(v1)):
            aIntob += v1[i] * v2[i]
            aSquare += v1[i] * v1[i]
            bSquare += v2[i] * v2[i]

    return aIntob / (mt.sqrt(aSquare) * mt.sqrt(bSquare))

def CencosineDeff(v1, v2, r1, r2):
    aIntob = 0
    aSquare = 0
    bSquare = 0
    newV1 = 0
    newV2 = 0

    if(len(v1) == len(v2)):
        for i in range(len(v1)):
            if (v1[i] != 0):
                newV1 = v1[i]-(r1)
            else:
                newV1 = 0

            if (v2[i] != 0):
                newV2 = v2[i]-(r2)
            else:
                newV2 = 0

            aIntob += newV1 * newV2
            aSquare += newV1 * newV1
            bSquare += newV2 * newV2

    return aIntob / (mt.sqrt(aSquare) * mt.sqrt(bSquare))

def pearsonCorr(v1, v2):

    sum_a = 0
    sum_b = 0
    aIntob = 0
    aSquare = 0
    bSquare = 0
    n = len(v1)
    r = 0

    if(len(v1) == len(v2)):
        for i in range(len(v1)):
            sum_a += v1[i]
            sum_b += v2[i]

            aIntob += v1[i] * v2[i]
            aSquare += v1[i] * v1[i]
            bSquare += v2[i] * v2[i]

    numerator = (n*(aIntob)-(sum_a*sum_b))
    denumerator = mt.sqrt(((n*aSquare)-(sum_a*sum_a))
                          * ((n*bSquare)-(sum_b*sum_b)))

    r = numerator/denumerator
    return r

def listAverage(arr):
    sumN = 0
    lessen = 0
    for n in range(len(arr)):
        if(arr[n] != 0):
            sumN += arr[n]
        else:
            lessen += 1
    return sumN / (len(arr) - lessen)

def CenteredCosimMatrix(arr):
    conresult = []

    for i in range(len(arr)):
        conresult.append([])

    for i in range(len(arr)):
        r1 = listAverage(arr[i])

        for k in range(i, len(arr)):
            r2 = listAverage(arr[k])

            r3 = CencosineDeff(arr[i], arr[k], r1, r2)

            conresult[i].append(r3)
            if(i != k):
                conresult[k].append(r3)
    return conresult

def AvgCossimMatrix(arr):
    cenresult = []
    for i in arr:
        cenresult.append([])

    for i in range(len(arr)):
        avg = listAverage(arr[i])

        for j in range(i, len(arr)):
            avg0 = listAverage(arr[j])
            arr1 = copy.deepcopy(arr[i])
            arr2 = copy.deepcopy(arr[j])
            for x in range(len(arr)):
                if(mt.isnan(arr1[x])):
                    arr1[x] = avg

                if(mt.isnan(arr2[x])):
                    arr2[x] = avg0

            r = AvgCossimDeff(arr1, arr2)
            cenresult[i].append(r)
            if(i != j):
                cenresult[j].append(r)
    return cenresult

def PearMatrix(arr):
    peresult = []
    for i in range(len(arr)):
        peresult.append([])
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            r = pearsonCorr(arr[i], arr[j])
            peresult[i].append(r)
            if(i != j):
                peresult[j].append(r)
    return peresult

def simFunc(arr1, arr2):
    result = cosineDeff(arr1, arr2)
    return result

def simMatrix(arr):
    result = []
    for i in range(len(arr)):
        result.append([])
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            r = simFunc(arr[i], arr[j])
            result[i].append(r)
            if(i != j):
                result[j].append(r)
    return result


# In[82]:


result = simMatrix(arr)
cenresult = AvgCossimMatrix(arr)
conresult = CenteredCosimMatrix(arr)
peresult = PearMatrix(arr)


# # Similarity Matrix

# In[61]:


print('')
print('cosine similarity Matrix: ')
print(np.array(result))
print('')
print('Avg Consin Matrix: ')
print(np.array(cenresult))
print('')
print('Centered Consin Matrix: ')
print(np.array(conresult))
print('')
print('Pearson Matrix: ')
print(np.array(peresult))
print('')


# In[62]:


def average2(v1, v2):
    return (v1 + v2) / 2


def average3(v1, v2, v3):
    return (v1 + v2 + v3) / 3


def average4(v1, v2, v3, v4):
    return (v1 + v2 + v3 + v4) / 4


# In[63]:


result = simMatrix(arr)


# In[84]:


def sortedMatrix(result):
    rs = []
    for i in result:
        i.sort(reverse=True)
        rs.append(i)
    return rs


# # Sorted Indices Matrix

# In[65]:


key = []
for i in result:
    key.append(sorted(range(len(i)), reverse=True, key=lambda k: i[k]))
key = np.array(key)

pkey = []
for i in peresult:
    pkey.append(sorted(range(len(i)), reverse=True, key=lambda k: i[k]))
pkey = np.array(pkey)


# In[66]:


print('Indices Cosine Matrix: ')
print(key)
print('')

print('Indices Pearson Matrix: ')
print(pkey)
print('')


# In[67]:


sortedArr = sortedMatrix(result)
sortedArrP = sortedMatrix(peresult)

sortedArr2 = np.array(sortedArr)
sortedArrP2 = np.array(sortedArrP)


# In[68]:


print('Similarity Indices Cosine Matrix: ')
print(sortedArr2)
print("")
print('Similarity Indices Pearson Matrix: ')
print(sortedArrP2)


# # Weighted Average

# In[69]:


def wAverage(user, num, pridictedIndex):
    similerUser = {}
    for i in range(len(key)):
        if((i + 1) <= num):
            similerUser.update(
                {key[int(user)][i + 1]: sortedArr2[int(user)][i]})
    s = 0
    avgTotal = 0
    weightedAverage = 0

    index = pridictedIndex
    for similerUserKeys in similerUser:
        s += similerUser[similerUserKeys] * arr[similerUserKeys][int(index)]
        avgTotal += similerUser[similerUserKeys]
    weightedAverage = s/avgTotal

    return weightedAverage


def wAverageI(user, num, pridictedIndex):
    similerUser = {}
    for i in range(len(key)):
        if((i + 1) <= num):
            similerUser.update(
                {key[int(user)][i + 1]: sortedArr3[int(user)][i]})
    s = 0
    avgTotal = 0
    weightedAverage = 0

    index = pridictedIndex
    for similerUserKeys in similerUser:
        s += similerUser[similerUserKeys] * arr[similerUserKeys][int(index)]
        avgTotal += similerUser[similerUserKeys]
    weightedAverage = s/avgTotal

    return weightedAverage

def pwAverage(user, num, pridictedIndex):
    similerUser = {}
    for i in range(len(pkey)):
        if((i + 1) <= num):
            similerUser.update(
                {pkey[int(user)][i + 1]: sortedArrP2[int(user)][i]})
    s = 0
    avgTotal = 0
    weightedAverage = 0

    index = pridictedIndex
    for similerUserKeys in similerUser:
        s += similerUser[similerUserKeys] * arr[similerUserKeys][int(index)]
        avgTotal += similerUser[similerUserKeys]
    weightedAverage = s/avgTotal

    return weightedAverage


def pwAverageI(user, num, pridictedIndex):
    similerUser = {}
    for i in range(len(pkey)):
        if((i + 1) <= num):
            similerUser.update(
                {pkey[int(user)][i + 1]: sortedArrP2[int(user)][i]})
    s = 0
    avgTotal = 0
    weightedAverage = 0

    index = pridictedIndex
    for similerUserKeys in similerUser:
        s += similerUser[similerUserKeys] * arr[similerUserKeys][int(index)]
        avgTotal += similerUser[similerUserKeys]
    weightedAverage = s/avgTotal

    return weightedAverage


# In[70]:


def replaceWithW(arr, value, predictedValue):
    arr[predictedValue] = value


# In[71]:


arrCopy = copy.deepcopy(arr)
arrCopy2 = copy.deepcopy(arr)
arrcopym = copy.deepcopy(arr)

def newModel (arr1, arr2):
    i = 0
    for i in range(len(arr1)):
        for j in range(len(arr1)):    
            arrcopym[i][j] = ((arr1[i][j]+arr2[i][j])/2)
    
    return arrcopym


# # User to User

# 4 Similar Users

# In[89]:


#Cosine Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')
print('\n[4] Most Similar Users to User (Cosine method)[' + user+'] are:')
avg = average4(sortedArr2[int(user)][1],
               sortedArr2[int(user)][2], sortedArr2[int(user)][3], sortedArr[int(user)][4])
replaceWithW(arrCopy[int(user)], float(wAverage(user, 4, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(key)):
    if((i + 1) <= 4):
        print('User[' + str(key[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(wAverage(user, 4, pridictedValue))))

#Pearson Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')
print('\n[4] Most Similar Users to User (Pearson method)[' + user+'] are: ')
avg = average4(sortedArrP2[int(user)][1],
               sortedArrP2[int(user)][2], sortedArrP2[int(user)][3], sortedArrP2[int(user)][4])
replaceWithW(arrCopy2[int(user)], float(pwAverage(user, 4, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy2[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(pkey)):
    if((i + 1) <= 4):
        print('User[' + str(pkey[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(pwAverage(user, 4, pridictedValue))))

print("New model based after averaging Cosine and Pearson:")
mse = sklearn.metrics.mean_squared_error(arrCopy, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy, arrCopy2))
print("")
print("Original:")
print(arrCopy)


# 3 Similar Users

# In[73]:


#Cosine Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')

print('\n[3] Most Similar Users to User (Cosine Method) [' + user+'] are:')
avg = average3(sortedArr2[int(user)][1],
               sortedArr2[int(user)][2], sortedArr2[int(user)][3])
replaceWithW(arrCopy[int(user)], float(wAverage(user, 3, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(key)):
    if((i + 1) <= 3):
        print('User[' + str(key[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(wAverage(user, 3, pridictedValue))))

#Pearson Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')
print('\n[3] Most Similar Users to User (Pearson method) [' + user+'] are: ')
avg = average3(sortedArrP2[int(user)][1],
               sortedArrP2[int(user)][2], sortedArrP2[int(user)][3])
replaceWithW(arrCopy2[int(user)], float(pwAverage(user, 3, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy2[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(pkey)):
    if((i + 1) <= 3):
        print('User[' + str(pkey[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(pwAverage(user, 3, pridictedValue))))

print("New model based after averaging Cosine and Pearson: ")
mse = sklearn.metrics.mean_squared_error(arrCopy, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy, arrCopy2))
print("")
print("Original: ")
print(arrCopy)


# 2 Similar Users

# In[74]:


#Cosine Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')

print('\n[2] Most Similar Users to User (Cosine Method) [' + user+'] are:')
avg = average2(sortedArr2[int(user)][1],
               sortedArr2[int(user)][2])
replaceWithW(arrCopy[int(user)], float(wAverage(user, 2, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(key)):
    if((i + 1) <= 2):
        print('User[' + str(key[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(wAverage(user, 2, pridictedValue))))

#Pearson Method
user = input('Specify the User to reveal similar Users: ')
pridictedValue = input('Specify the index to pridict: ')
print('\n[2] Most Similar Users to User (Pearson method) [' + user+'] are: ')
avg = average2(sortedArrP2[int(user)][1],
               sortedArrP2[int(user)][2])
replaceWithW(arrCopy2[int(user)], float(pwAverage(user, 2, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(arr[int(user)], arrCopy2[int(user)])
rmse = mt.sqrt(mse)
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')

for i in range(len(pkey)):
    if((i + 1) <= 2):
        print('User[' + str(pkey[int(user)][i + 1])+']')

print('\n[Pridicted missing value through Weighted Average]: ' +
      str(float(pwAverage(user, 2, pridictedValue))))

print("New model based after averaging Cosine and Pearson: ")
mse = sklearn.metrics.mean_squared_error(arrCopy, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy, arrCopy2))
print("")
print("Original: ")
print(arrCopy)


# # Item - Item

# In[85]:


tranposedArray = arr.T

print(np.array(tranposedArray))

result2 = []
result2 = simMatrix(arrT)

presult2 = []
presult2 = PearMatrix(arrT)

print('\n Item to item cosine /n ' + str(np.array(result2)))

print('\n item to item pearson /n' + str(np.array(presult2)))


# # Transposed Sorted Indices Matrix

# In[76]:


key2 = []
for i in result2:
    key2.append(sorted(range(len(i)), reverse=True, key=lambda k: i[k]))
key2 = np.array(key2)

pkey2 = []
for i in presult2:
    pkey2.append(sorted(range(len(i)), reverse=True, key=lambda k: i[k]))
pkey2 = np.array(pkey2)


# In[77]:


print("")
print(key2)
print("")
print(pkey2)


# In[78]:


arrCopy2 = copy.deepcopy(tranposedArray)
parrCopy2 = copy.deepcopy(tranposedArray)


# 4 Similar Items

# In[79]:


#Cosine Method
sortedArr3 = sortedMatrix(result2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('specify the index to pridict: ')

avg = average4(sortedArr3[int(item)][1], sortedArr3[int(
    item)][2], sortedArr3[int(item)][3], sortedArr3[int(item)][4])
replaceWithW(arrCopy2[int(item)], float(wAverageI(item, 4, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], arrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[4] Most Similar Items to Item (Cosine Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 4):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(wAverageI(item, 4, pridictedValue))))

#Pearson Method
sortedArrP3 = sortedMatrix(presult2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('Specify the index to pridict: ')

avg = average4(sortedArrP3[int(item)][1], sortedArrP3[int(
    item)][2], sortedArrP3[int(item)][3], sortedArrP3[int(item)][4])
replaceWithW(parrCopy2[int(item)], float(pwAverageI(item, 4, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], parrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[4] Most Similar Items to Item (Pearson Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 4):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(pwAverageI(item, 4, pridictedValue))))

print("New model based after averaging Cosine and Pearson: ")
mse = sklearn.metrics.mean_squared_error(arrCopy, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy2, parrCopy2))
print("")
print("Original: ")
print(arrCopy2)


# 3 Similar Items

# In[80]:


#Cosine Method
sortedArr3 = sortedMatrix(result2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('Specify the index to pridict: ')

avg = average3(sortedArr3[int(item)][1], sortedArr3[int(
    item)][2], sortedArr3[int(item)][3])
replaceWithW(arrCopy2[int(item)], float(wAverageI(item, 3, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], arrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[3] Most Similar Items to Item (Cosine Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 3):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(wAverageI(item, 3, pridictedValue))))

#Pearson Method
sortedArrP3 = sortedMatrix(presult2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('Specify the index to pridict: ')

avg = average3(sortedArrP3[int(item)][1], sortedArrP3[int(
    item)][2], sortedArrP3[int(item)][3])
replaceWithW(parrCopy2[int(item)], float(pwAverageI(item, 3, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], parrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[3] Most Similar Items to Item (Pearson Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 3):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(pwAverageI(item, 3, pridictedValue))))

print("New model based after averaging Cosine and Pearson: ")
mse = sklearn.metrics.mean_squared_error(parrCopy2, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy2, parrCopy2))
print("")
print("Original: ")
print(arrCopy2)


# 2 Similar items

# In[88]:


#Cosine Method
sortedArr3 = sortedMatrix(result2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('Specify the index to pridict: ')

avg = average2(sortedArr3[int(item)][1], sortedArr3[int(
    item)][2])
replaceWithW(arrCopy2[int(item)], float(wAverageI(item, 2, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], arrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[2] Most Similar Items to Item (Cosine Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 2):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(wAverageI(item, 2, pridictedValue))))

#Pearson Method
sortedArrP3 = sortedMatrix(presult2)
item = input('Specify the Item to reveal similar Item: ')
pridictedValue = input('Specify the index to pridict: ')

avg = average2(sortedArrP3[int(item)][1], sortedArrP3[int(
    item)][2])
replaceWithW(parrCopy2[int(item)], float(pwAverageI(item, 2, pridictedValue)), int(pridictedValue))
mse = sklearn.metrics.mean_squared_error(
    tranposedArray[int(item)], parrCopy2[int(item)])
rmse = mt.sqrt(mse)
print('[3] Most Similar Items to Item (Pearson Method) ['+item+'] are:')
print('[Average]: '+str(float(avg)))
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
for i in range(len(key2)):
    if((i + 1) <= 2):
        print('Item[' + str(key2[int(item)][i + 1])+']')

print('\n[Weighted Average]: '+str(float(pwAverageI(item, 2, pridictedValue))))
print("New model based after averaging Cosine and Pearson: ")
mse = sklearn.metrics.mean_squared_error(parrCopy2, arrCopy2)
rmse = mt.sqrt(mse)
print('[Accuracy]:'+str(100 - rmse)+' RMSE')
print(newModel(arrCopy2, parrCopy2))
print("")
print("Original: ")
print(arrCopy2)


# In[ ]:




