import os
import cv2
import numpy as np



class Imp:


    """
    Cette classe a pour but d'importer les images et de les transformer en matrices.
    la fonction imp() pour les images de test et de train
    la fonction imp_perso() pour les images d'évaluation

    """


    def __init__(self):
        self.data_train, self.data_test, self.data_val = [], [], []
        self.target_train, self.target_test, self.target_val = [], [], []



    def imp(self):
        # importer le nom des images des données de train
        for i in range(len(os.listdir('./mninst/train'))):
            for j in (os.listdir(f'./mninst/train/{str(i)}')):
                
                # récupérer la target (nom du dossier) dans une liste
                self.target_train.append(i)
                
                
                # récupérer le chemin de l'image 
                path = f'./mninst/train/{str(i)}/{j}'
                
                # convertir l'image en np.array ; la lire en niveau de gris
                im = cv2.imread(path, 0)
                # mettre les images à une dimension et changer la valeurs des pixels
                im = im.ravel()
                
                im[im<51] = 0
                im[im>50] = 255
                # im = im.reshape((28,28,1))
                
                # mettre ce np.array dans une liste
                self.data_train.append(im)


        
        # importer le nom des images des données de test
        for i in range(len(os.listdir('./mninst/test'))):
            for j in (os.listdir(f'./mninst/test/{str(i)}')):
                
                # récupérer la target (nom du dossier) dans une liste
                self.target_test.append(i)
                
                # récupérer le chemin de l'image 
                path = f'./mninst/test/{str(i)}/{j}'
                
                # convertir l'image en np.array ; la lire en niveau de gris
                im = cv2.imread(path, 0)
                im = im.ravel()
                im[im<51] = 0
                im[im>50] = 255
                
                # im = im.reshape((28,28,1))
               
        
                # mettre ce np.array dans une liste
                self.data_test.append(im)


        X_train = np.array(self.data_train)
        X_test = np.array(self.data_test)

       
        y_train = np.array(self.target_train)
        y_test = np.array(self.target_test)


       
        return X_train, y_train, X_test, y_test
    




    def imp_perso(self):

        """
        importer les fichiers créés pour évaluer le modèle 
        """


        for i in range(len(os.listdir('./perso'))):
            for j in (os.listdir(f'./perso/{str(i)}')):
                
                # récupérer la target (nom du dossier) dans une liste
                self.target_val.append(i)         
                
                # récupérer le chemin de l'image 
                path = f'./perso/{str(i)}/{j}'
                
                # convertir l'image en np.array ; la lire en niveau de gris
                im = cv2.imread(path, 0)
                # mettre les images à une dimension et changer la valeurs des pixels
                im = im.ravel()
                
                im[im<51] = 0
                im[im>50] = 255
                # im = im.reshape((28,28,1))
                # mettre ce np.array dans une liste
                self.data_val.append(im)

        X_val = np.array(self.data_val)
        y_val = np.array(self.target_val)

        return X_val, y_val



if __name__=='__main__':

    cls = Imp()

    X_train, y_train, X_test, y_test = cls.imp()
    X_val, y_val = cls.imp_perso()

    print(f'Taille des données d\'entraînement : {X_train.shape}')
    print(f'Taille de la cible d\'entraînement : {y_train.shape}')
    print(f'Taille des données de test : {X_test.shape}')
    print(f'Taille de la cible de test : {y_test.shape}')
    print(f'Taille des données d\'entraînement : {X_val.shape}')
    print(f'Taille de la cible d\'entraînement : {y_val.shape}')
