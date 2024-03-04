#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:27:56 2021

@author: fehrenbach

TEST DU FAST MARCHING

"""

import matplotlib.pyplot as plt
import numpy as np
import time
import imageio
import heapq

def voisins(i, j, nx, ny) :
    '''
    Cette fonction renvoie les indices des pixels voisins du pixel (i, j) 
    dans une grille de taille nx par ny

    Parameters
    ----------
    i : int
        L'indice de la ligne du pixel dans la grille
    j : int
        L'indice de la colonne du pixel dans la grille
    nx : int
        La taille de la grille selon l'axe horizontal
    ny : int
        La taille de la grille selon l'axe vertical

    Returns
    -------
    listeVoisins : Liste de Tuples
        Une liste contenant les indices des pixels voisins valides du pixel (i, j)
    '''
    listeVoisins = []
    
    # Liste de Tuples correpondants aux déplacements pour trouver les voisins
    positionVoisins = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for vx, vy in positionVoisins :
        voisinI, voisinJ = i + vx, j + vy
        
        # On vérifie si les voisins peuvent être dans la grille
        if ( ( 0 <= voisinI < nx )  and ( 0 <= voisinJ < ny ) ) :
            listeVoisins.append((voisinI, voisinJ))
            
    return listeVoisins

def dijkstraSurGrille(k, s):
    ny = len(k[0])
    nx = len(k)
    
    sx = s[0] - 1
    sy = s[1] - 1
    
    tag = [[0 for _ in range(ny)] for _ in range(nx)] # 0 = Eloigné, -1 = Accepté, 1 = Front
    d = [[float('inf') for _ in range(ny)] for _ in range(nx)]
    predx = [[-1 for _ in range(ny)] for _ in range(nx)]
    predy = [[-1 for _ in range(ny)] for _ in range(nx)]
    H = []  # Tas vide qui servira à stocker le front : n-uples (distance, sommetx, sommety, prédécesseurx, prédécesseury)
    tag[sx][sy] = -1
    d[sx][sy] = 0
    predx[sx][sy] = -1
    predy[sx][sy] = -1
    for v in voisins(s[0], s[1], nx, ny):
        tag[v[0]][v[1]] = 1
        heapq.heappush(H, (k[v[0]][v[1]], v[0], v[1], s[0], s[1]))
    while len(H) > 0:
        minH = heapq.heappop(H)
        dMin = minH[0]
        vMinX = minH[1]
        vMinY = minH[2]
        pMinX = minH[3]
        pMinY = minH[4]
        if tag[vMinX][vMinY] != -1:
            d[vMinX][vMinY] = dMin
            tag[vMinX][vMinY] = -1
            predx[vMinX][vMinY] = pMinX
            predy[vMinX][vMinY] = pMinY
            for v in voisins(vMinX, vMinY, nx, ny):
                tag[v[0]][v[1]] = 1
                if d[v[0]][v[1]] > d[vMinX][vMinY] + k[v[0]][v[1]]:
                    d[v[0]][v[1]] = d[vMinX][vMinY] + k[v[0]][v[1]]
                    heapq.heappush(H, (d[vMinX][vMinY] + k[v[0]][v[1]], v[0], v[1], vMinX, vMinY))
    return d, predx, predy

nx = 200
ny = 250


### test 1
#W = np.ones((nx,ny))
x0=[100,10]
target = [100,190]
##
# test 2
x = np.arange(nx)-0.5*nx
y = np.arange(ny)-0.5*nx
sigma = 10
xg, yg = np.meshgrid(x,y)
## cas 1
W = np.ones((nx,ny))
## cas 2
#W = np.ones((nx,ny))
W[np.where(xg**2+yg**2<20**2)] = 10
## cas 3 labyrinthe
##u = imageio.imread('labyrinthe002.jpg')
#uu = np.sum(u,axis=2)
#nx = np.shape(uu)[0]
#ny = np.shape(uu)[1]
#W = np.ones((nx,ny))
#W[np.where(uu<300)] = 500
#x0=[45,10]
#target = [208,270]




plt.figure(2)
plt.imshow(W)
#%%
t1 = time.time()
d,px,py,t = dijkstraSurGrille(W,x0)
t2 = time.time()
chrono = t2-t1
print(' --- temps : ',chrono)

px = np.array(px)  # Conversion en tableau numpy
py = np.array(py)  # Conversion en tableau numpy
      
plt.figure(1)
plt.contourf(d,20,cmap='jet')
plt.colorbar()
plt.contour(d,20,colors='k')
#%%

plt.figure(3)
plt.imshow(W)
### descente
scourant = target
while not(scourant==x0):
    plt.plot(scourant[1],scourant[0],'r.')
    scourant = [px[scourant[0],scourant[1]],py[scourant[0],scourant[1]]]

