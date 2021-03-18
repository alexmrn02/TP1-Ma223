# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import time


##################################### GAUSSS ####################################################


def ReductionGauss (Aaug) : 
    n,m = np.shape(Aaug)
    for k in range (0,n-1):
        for i in range(k+1,n):
            #print(Aaug)
            gik = Aaug[i,k]/Aaug[k,k]
            Aaug[i, :]= Aaug[i, :]-(gik*Aaug[k, :])
            
    return Aaug
        



def ResolutionSystTriSup(Taug):
    n,m = Taug.shape
    if m!=n+1:
        return('pas une matrice augmentée')
    x=np.zeros(n)
    for i in range(n-1,-1,-1):
        somme=0
        for k in range(i+1,n):
            somme=somme+x[k]*Taug[i,k]
        x[i]=(Taug[i,-1]-somme)/Taug[i,i]
    return x


    
def Gauss(A,B):
    Aaug = np.c_[A,B]
    solution = ResolutionSystTriSup(ReductionGauss(Aaug))
    return solution

    
##################################### Décomposition/Résolution LU #############################


def DecompositionLU(A):
    n,m = A.shape 
    L = np.eye(n)
    U = np.copy(A)
    for i in range(0,n-1):
        for k in range(i+1,n):
            pivot = U[k,i]/U[i,i]
            L[k,i]= pivot
            for j in range(i,n):
                U[k,j]=U[k,j]-pivot*U[i,j]
    return L,U


def ResolutionSystTriInf(Taug):
    n,m = Taug.shape
    if m!=n+1:
        print('pas une matrice augmentée')
        return
    x=np.zeros(n)
    for i in range(n):
        somme=0
        for K in range(n):
            somme=somme+x[K]*Taug[i,K]
        x[i]=(Taug[i,-1]-somme)/Taug[i,i]
    return x

def ResolutionLU(A,b):
    n,m = np.shape(A)
    X=np.zeros(n)
    L,U=DecompositionLU(A)
    #print("L=\n", L)
    #print("U=\n", U)
    Y=ResolutionSystTriInf(np.c_[L,b])
    Y1=np.asarray(Y).reshape(n,1)
    X=ResolutionSystTriSup(np.c_[U,Y1])
    return (X)

######################################## GaussChoixPivotPartiel ##############################
    

def GaussChoixPivotPartiel(A,B):
    Aaug=np.c_[A,B]
    nb_ligne, nb_colonnes = np.shape(Aaug)
    L_echange=0
    pivot = 0
    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            if Aaug[k,k] == 0:
                plusgrand = 0
                for l in range (i, nb_ligne):
                    if abs(Aaug[l,k]) > plusgrand:
                        plusgrand = abs(Aaug[l,k])
                        L_echange = l
                c=Aaug[i-1,:].copy()
                Aaug[i-1,:] = Aaug[L_echange,:]
                Aaug[L_echange,:]= c
            pivot = (Aaug[i,k])/(Aaug[k,k])

            Aaug[i,:] = Aaug[i,:] - pivot * Aaug[k,:]
    solution = ResolutionSystTriSup(Aaug)
    return solution

##################################### GaussChoixPivotTotal ##############€#########
    

def GaussChoixPivotTotal(A,B):
    X=np.c_[A,B]
    n,m = X.shape
    if m!= n+1:
        print("erreur de format")
    else:
        for k in range(m-1):
            for i in range(k+1,m-1):
                for var in range(i,m-1):
                    if abs(X[var,k]) > abs(X[k,k]):
                        L0 = np.copy(X[k,:])
                        X[k,:] = X[var,:]
                        X[var,:] = L0
                if X[k,k] == 0:
                    print('erreur pivot nul')
                g = X[i,k]/X[k,k]
                X[i,:] = X[i,:] - g*X[k,:]
    
    X=ResolutionSystTriSup(X)

    return X
################################ Courbe de temps ##################################


Temps = []
Indices = []
TempsLU = []
IndicesLU = []
TempsGP = []
IndicesGP=[]
TempsGT = []
IndicesGT=[]


for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t1 = time.time()
        Gauss(A,B)
        t2 = time.time()
        t = t2 - t1
        Temps.append(t)
        Indices.append(n)
    except:
        print('')
        
        
for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t3 = time.time()
        X = ResolutionLU(A,B)
        t4 = time.time()
        T = t4 - t3
        TempsLU.append(T)
        IndicesLU.append(n)

    except:
        print('')
 
for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t5 = time.time()
        X = GaussChoixPivotPartiel(A,B)
        t6 = time.time()
        T1 = t6 - t5
        TempsGP.append(T1)
        IndicesGP.append(n)

    except:
        print('') 
       
for n in range(50,500,50):
    try:
        A= np.random.randint(low = 1,high = n,size = (n,n))
        B= np.random.randint(low = 1,high = n,size = (n,1))
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        t7 = time.time()
        X = GaussChoixPivotTotal(A,B)
        t8 = time.time()
        T2 = t8 - t7
        TempsGT.append(T2)
        IndicesGT.append(n)

    except:
        print('') 


x = Indices 
y = Temps
x1 = IndicesLU
y1 = TempsLU
x2 = IndicesGP
y2 = TempsGP
x3 = IndicesGT
y3 = TempsGT

plt.plot(x,y,color='red', label='Gauss')
plt.plot(x1,y1,color='green', label='LU')
plt.plot(x2,y2,color="blue", label='Pivot Partiel')
plt.plot(x3,y3,color="yellow", label='Pivot Total')
plt.xlabel('dimension')
plt.ylabel('temps(secondes)')
plt.title("Temps en fonction de la dimension de la matrice")
plt.legend()
plt.show()


############################### Courbes d'erreur ##########################################


Erreur = []
Indices = []
ErreurLU = []
IndicesLU = []
ErreurGP=[]
IndicesGP =[]
ErreurGT = []
IndicesGT=[]

for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        X = Gauss(A,B)
        err = np.linalg.norm(A.dot(X)-np.ravel(B))
        Erreur.append(err)
        Indices.append(n)

    except:
        print('')
        


for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        X = ResolutionLU(A,B)
        err1 = np.linalg.norm(A.dot(X)-np.ravel(B))
        ErreurLU.append(err1)
        IndicesLU.append(n)
        

    except:
        print('')

      
for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        X = GaussChoixPivotPartiel(A,B)
        err2 = np.linalg.norm(A.dot(X)-np.ravel(B))
        ErreurGP.append(err2)
        IndicesGP.append(n)

    except:
        print('')
        
for n in range(50,500,50):
    try:
        A= np.random.rand(n,n)
        B= np.random.rand(n)
        A = np.array(A, dtype = float)
        B = np.array(B, dtype = float)
        X = GaussChoixPivotTotal(A,B)
        err3 = np.linalg.norm(A.dot(X)-np.ravel(B))
        ErreurGT.append(err3)
        IndicesGT.append(n)

    except:
        print('')
      
xe = Indices
ye = Erreur
x1e = IndicesLU
y1e = ErreurLU
x2e = IndicesGP
y2e = ErreurGP
x3e = IndicesGT
y3e = ErreurGT
plt.plot(xe,ye,color='red', label='Gauss')
plt.plot(x1e,y1e,color='green', label='LU')
plt.plot(x2e,y2e,color='blue', label='Pivot Partiel')
plt.plot(x3e,y3e,color='yellow', label='Pivot Total')
plt.xlabel('dimension')
plt.ylabel('normes(erreurs)')
plt.title("Erreur en fonction de la dimension de la matrice")
plt.legend()
plt.show()
