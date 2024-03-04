# ---------- IMPORTS ---------- #

import heapq
import imageio
import numpy as np
from matplotlib import pyplot as plt

# ---------- FONCTIONS ---------- #

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


def dijkstraSurGrille(k, s) :
    '''
    Cette fonction implémente l'algorithme de Dijkstra sur une grille avec des coûts k.

    Parameters
    ----------
    k : Matrice - [[int]]
        Matrice de coûts représentant les poids des arêtes entre les sommets de la grille.
    s : Tuple(int, int)
        Coordonnées du sommet source s = (sx, sy).

    Returns
    -------
    d : Matrice - [[float]]
        Tableau des distances minimales depuis le sommet source vers chaque sommet de la grille.
    predx : Matrice - [[int]]
        Tableau contenant les indices x des prédécesseurs de chaque sommet vers la source.
    predy : Matrice - [[int]]
        Tableau contenant les indices y des prédécesseurs de chaque sommet vers la source.

    '''
    ny = len(k[0])  # Nombre de colonnes dans la matrice des coûts
    nx = len(k)     # Nombre de lignes dans la matrice des coûts
    
    sx = s[0]       # Coordonnée x du sommet source
    sy = s[1]       # Coordonnée y du sommet source
    
    # Initialisation des tableaux
    tag = [[0 for _ in range(ny)] for _ in range(nx)] # 0 = Eloigné, -1 = Accepté, 1 = Front
    d = [[float('inf') for _ in range(ny)] for _ in range(nx)]  # Tableau des distances
    predx = [[-1 for _ in range(ny)] for _ in range(nx)]        # Indices x des prédécesseurs
    predy = [[-1 for _ in range(ny)] for _ in range(nx)]        # Indices y des prédécesseurs
    H = []  # Tas vide qui servira à stocker le front : n-uples (distance, sommetx, sommety, prédécesseurx, prédécesseury)
    tag[sx][sy] = -1    # Marquer le sommet source comme accepté
    d[sx][sy] = 0       # Distance du sommet source à lui-même est 0
    predx[sx][sy] = -1  # Aucun prédécesseur pour le sommet source en x
    predy[sx][sy] = -1  # Aucun prédécesseur pour le sommet source en y
    
    # Initialiser les voisins du sommet source
    for v in voisins(s[0], s[1], nx, ny) :
        tag[v[0]][v[1]] = 1
        heapq.heappush(H, (k[v[0]][v[1]], v[0], v[1], s[0], s[1]))
       
    # Algorithme principal de Dijkstra
    while len(H) > 0:
        dMin, vMinX, vMinY, pMinX, pMinY = heapq.heappop(H)
        if tag[vMinX][vMinY] != -1 :
            d[vMinX][vMinY] = dMin
            tag[vMinX][vMinY] = -1
            predx[vMinX][vMinY] = pMinX
            predy[vMinX][vMinY] = pMinY
            for v in voisins(vMinX, vMinY, nx, ny) :
                tag[v[0]][v[1]] = 1
                if d[v[0]][v[1]] > d[vMinX][vMinY] + k[v[0]][v[1]] :
                    d[v[0]][v[1]] = d[vMinX][vMinY] + k[v[0]][v[1]]
                    heapq.heappush(H, (d[vMinX][vMinY] + k[v[0]][v[1]], v[0], v[1], vMinX, vMinY))
    return d, predx, predy                

# ---------- CODE ---------- #

# Question 3

u = imageio.imread('labyrinthe002.jpg')
uu = np.sum(u,axis=2)
nx = np.shape(uu)[0]
ny = np.shape(uu)[1]
K = np.ones((nx,ny))
K[np.where(uu<300)] = 500
x0=[45,10]
d, px, py = dijkstraSurGrille(K, x0)
plt.figure(1)
plt.contourf(d,20)

# Question 4

plt.figure(3)
plt.imshow(K)
px = np.array(px)
py = np.array(py)
scourant = [208,270]
while not(scourant==x0):
    plt.plot(scourant[1],scourant[0],'r.')
    scourant = [px[scourant[0],scourant[1]],py[scourant[0],scourant[1]]]
