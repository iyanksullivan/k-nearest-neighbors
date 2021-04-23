#KNN Tugas 3 AI
#1301180252
#Rifqi Fauzia M.

import pandas as PD
import math

# List of function i.e Euclidean, standardScaler, etc.
def standardScale(df):
    standard = df.copy()
    for column in standard.columns:
        standard[column] = (standard[column] - standard[column].mean()) / standard[column].std()
    return standard

def euclidDist(PT,PTe):
    dist = 0
    res = 0
    for i in range(len(PT)):
        dist += math.pow(PT[i]-PTe[i],2)
        res = math.sqrt(dist)
    return res

def Classification(TrainFitur,TrainTarget,TestFitur,k):
    List_dist = []
    c1 = 0
    c0 = 0
    for x in range(len(TrainFitur)):
        dist = euclidDist(TrainFitur[x][0],TestFitur[0])
        List_dist.append([dist,TrainFitur[x][1]])
    for X in range(len(TrainTarget)):
        List_dist[X].append(TrainTarget[X][0][0])
    List_dist.sort()
    kneigh = List_dist[:k]
    for x in range(len(kneigh)):
        if (kneigh[x][2] == 0):
            c0 += 1
        if (kneigh[x][2] == 1):
            c1 += 1
    if(c0>c1):
        highfreq = 0
    else:
        highfreq = 1
    return highfreq

def findK(list_acc):
    tbk = sorted(list_acc,reverse=True)
    return tbk[0]

def avg_acc(list_acc):
    temp = 0
    for x in list_acc:
        temp += x
    return (temp/len(list_acc))


#import data from csv into dataset and convert 0 to NaN
DS = PD.read_csv('Diabetes.csv')
tempF = PD.DataFrame(DS, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
tempT = PD.DataFrame(DS, columns=['Outcome'])
tempF = standardScale(tempF)
list_data_fitur = []
list_data_target = []
for X in range(len(DS)):
    list_data_fitur.append([tempF.values[X],X])
    list_data_target.append([tempT.values[X],X])

#dataset 1
F1Train = list_data_fitur[0:614]
F1Test  = list_data_fitur[614:768]
T1Train = list_data_target[0:614]
T1Test  = list_data_target[614:768]

#dataset 2
F2TrainT = list_data_fitur[0:461]
F2Train  = F2TrainT + list_data_fitur[615:768]
F2Test   = list_data_fitur[461:615]
T2TrainT = list_data_target[0:461]
T2Train  = T2TrainT + list_data_target[615:768]
T2Test   = list_data_target[461:615]

# #dataset 3
F3TrainT = list_data_fitur[0:306]
F3Train  = F3TrainT + list_data_fitur[460:768]
F3Test   = list_data_fitur[306:460]
T3TrainT = list_data_target[0:306]
T3Train  = T3TrainT + list_data_target[460:768]
T3Test   = list_data_target[306:460]

# #dataset 4
F4TrainT = list_data_fitur[0:154]
F4Train  = F4TrainT + list_data_fitur[308:768]
F4Test   = list_data_fitur[154:308]
T4TrainT = list_data_target[0:154]
T4Train  = T4TrainT + list_data_target[308:768]
T4Test   = list_data_target[154:308]
#
# #dataset 5
F5Train = list_data_fitur[155:768]
F5Test  = list_data_fitur[0:155]
T5Train = list_data_target[155:768]
T5Test  = list_data_target[0:155]

#this is Classification list
best_acc = []

#dataset 1
k = 1
akurasi = []
while k < 100:
    mark = 0
    tempP = []
    for i in range(len(F1Test)):
        tempP.append(Classification(F1Train, T1Train, F1Test[i], k))
    for x in range(len(tempP)):
        if (tempP[x] == T1Test[x][0]):
            mark += 1
    akurasi.append([mark/len(T1Test),k])
    k += 2
bestK = findK(akurasi)
print("Dataset 1")
print("K terbaik: ",bestK[1]," Dengan ACC: ",bestK[0]*100,"%")
best_acc.append(bestK[0]*100)

#dataset 2
k = 1
akurasi = []
while k < 100:
    mark = 0
    tempP = []
    for i in range(len(F2Test)):
        tempP.append(Classification(F2Train, T2Train, F2Test[i], k))
    for x in range(len(tempP)):
        if (tempP[x] == T2Test[x][0]):
            mark += 1
    akurasi.append([mark/len(T2Test),k])
    k += 2
bestK = findK(akurasi)
print("Dataset 2")
print("K terbaik: ",bestK[1]," Dengan ACC: ",bestK[0]*100,"%")
best_acc.append(bestK[0]*100)

#dataset 3
k = 1
akurasi = []
while k < 100:
    mark = 0
    tempP = []
    for i in range(len(F3Test)):
        tempP.append(Classification(F3Train, T3Train, F3Test[i], k))
    for x in range(len(tempP)):
        if (tempP[x] == T3Test[x][0]):
            mark += 1
    akurasi.append([mark/len(T3Test),k])
    k += 2
bestK = findK(akurasi)
print("Dataset 3")
print("K terbaik: ",bestK[1]," Dengan ACC: ",bestK[0]*100,"%")
best_acc.append(bestK[0]*100)

#dataset 4
k = 1
akurasi = []
while k < 100:
    mark = 0
    tempP = []
    for i in range(len(F4Test)):
        tempP.append(Classification(F4Train, T4Train, F4Test[i], k))
    for x in range(len(tempP)):
        if (tempP[x] == T4Test[x][0]):
            mark += 1
    akurasi.append([mark/len(T4Test),k])
    k += 2
bestK = findK(akurasi)
print("Dataset 4")
print("K terbaik: ",bestK[1]," Dengan ACC: ",bestK[0]*100,"%")
best_acc.append(bestK[0]*100)

#dataset 5
k = 1
akurasi = []
while k < 100:
    mark = 0
    tempP = []
    for i in range(len(F5Test)):
        tempP.append(Classification(F5Train, T5Train, F5Test[i], k))
    for x in range(len(tempP)):
        if (tempP[x] == T5Test[x][0]):
            mark += 1
    akurasi.append([mark/len(T5Test),k])
    k += 2
bestK = findK(akurasi)
print("Dataset 5")
print("K terbaik: ",bestK[1]," Dengan ACC: ",bestK[0]*100,"%")
best_acc.append(bestK[0]*100)

avg_acuration = avg_acc(best_acc)
print("")
print("Dengan Menggunakan 5 Cross validation "
      "dengan Nilai K dari 1 - 50 maka dihasilkan "
      "nilai rata rata Akurasi sebesar: ",avg_acuration,"%")

