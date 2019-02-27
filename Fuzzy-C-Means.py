import cv2
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import cv2

class Fuzzy:
    def __init__(self, img,nbrcluster,epsilon):
        self.img=img
        self.nbrCluster=nbrcluster
        self.centers=[]
        self.tab=None
        self.l, self.c = self.img.shape
        self.U = np.zeros((self.nbrCluster, self.l * self.c), dtype=np.float)
        self.UP= np.zeros((self.nbrCluster, self.l * self.c), dtype=np.float)
        self.img1=self.img.ravel()
        self.epsilon=epsilon
        self.resultatSegmenter=np.zeros(self.l *self.c )

    def initCluster(self):
        l, c = self.img.shape
        for i in range(self.nbrCluster):
            self.centers.append(random.randint(0,255))


    # le m est le parametre pour la force attration
    def degresAppartenance(self,m):
        for i in range(self.nbrCluster):
            for j in range(len(self.img1)):
                somme=0
                for k in range(self.nbrCluster):
                    somme = somme+(np.abs(self.img1[j]-self.centers[i]))/np.abs((self.img1[j]-self.centers[k]))

                self.U[i,j]=(1/ somme**(2/(m-1)))
        #print self.UP
        #print self.U

    def updateCenter(self, m):
        for i in range(len(self.centers)):
            self.centers[i]=sum((self.U[i,:]**m)*self.img1)/sum(self.U[i,:]**m)

    def norm(self):
        for i in range(self.nbrCluster):
            for j in range(len(self.img1.shape)):
                if self.U[i,j]-self.UP[i,j] > self.epsilon:
                    return False
        return True

    def lauch(self,m):
        self.initCluster()
        self.degresAppartenance(m)
        self.updateCenter(m)
        print self.norm()
        while (self.norm() == False):
            print " je met a jour "
            for i in range(self.nbrCluster):
                for j in range(len(self.img1.shape)):
                    self.UP[i,j]=self.U[i,j]
            self.degresAppartenance(m)
            self.updateCenter(m)
        print "-----------Matrice U---------------"
        print self.U
        print "-----------centroide---------------"
        print self.centers

    def segmentation(self):
        for j in range(len(self.img1)):
            s=np.argsort(self.U[:,j].ravel())[self.nbrCluster-1]
            self.resultatSegmenter[j]=self.centers[s]
            #print self.centers[s]
        return self.resultatSegmenter.reshape((self.l,self.c)).astype(np.uint8)

    def affiche(self,cluster):
        print np.reshape(self.U[cluster,:],(self.l,self.c))
        return np.reshape(self.U[cluster,:],(self.l,self.c))


if __name__ == "__main__":
    path="blobs1.gif"
    img=plt.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=Fuzzy(gray,3, 0.1)
    f.lauch(3)
    a = f.segmentation()
    b = f.affiche(0)
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.subplot(1,2,2)
    plt.imshow(b,cmap="gray")
    plt.show()