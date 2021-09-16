#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 08:55:23 2021

@author: rtankoua
"""

#-------------------------------------------------------------------------------#
# Module des fonctions auxillaires utilisées dans le script principal           #
# Importé dans le Pneumonie_ImgClassification_FlaskxGCP.py                      #
# Date: 2021 09                                                                 #
#-------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------#
# Import des modules
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import time, datetime, os, stat, glob, shutil, itertools
import itertools
from pylab import rcParams
from os.path import dirname as up
from PIL import Image
from shutil import copyfile
from datetime import date


#-------------------------------------------------------------------------------#
# Afficher les images brutes
def plots(ims, figsize=(14, 7.5), rows=1, interp=False, titles=None):
    '''
    Args:
        -- ims : liste des images (list)
        -- figsize : la taille souhaité des images (width and height)
        -- rows : nombre de rangés d'images souhaitées pour l'affichage par requête
    '''

    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=10)
        plt.imshow(ims[i].squeeze(), interpolation=None if interp else 'none')


#-------------------------------------------------------------------------------#

# S'il y a des problèmes causés par les repertoires innacessibles ou introuvables
def on_rm_error(func, path, exc_info):
    '''
    path donné en entré contient le chemin du fichier qui n'a pu être atteint
    On assume qu'il est juste lu et on va le "unlink" .
    '''

    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

    return None


#-------------------------------------------------------------------------------#
# Supprime les fichiers dans un dossier
def delete_files_in_folder(folder_path):
    '''
    Args:
        -- folder_path : un chemin donné du dossier qui contient les fichiers
    '''

    os.chmod(folder_path, 0o777)
    print('given folder address: {}'.format(folder_path))
    shutil.rmtree(folder_path, onerror = on_rm_error)
    print('folder deleted.')

    return None


#-------------------------------------------------------------------------------#
# Renomme le dossier par un nom de dossier donné
def rename_folder_with_single_class(current_folder_path, renamed_folder_path):
    '''
        -- current_folder_path : un chemin du dossier à renommer qui contient les fichiers   
        -- renamed_folder_path : nouveau chemin donné du dossier qui contient les fichiers   
    '''

    os.rename(current_folder_path, renamed_folder_path)
    print('Le répertoire pour une unique classe a été changé de \
    {} à {}'.format(current_folder_path, renamed_folder_path))
    
    return None


#-------------------------------------------------------------------------------#
# Vérifie si la classe abstraite dossier est dans un sous repertoire du projet
def abstract_class_exists(ABSTRACT_CLASS, l_DIRS):
    '''
        -- ABSTRACT_CLASS : nom de la classe abstraite qui doit être séparée.
        -- l_DIRS : listes des repertoires de train, test et val sets du projet.
    '''

    l_subs = []
    for folder in l_DIRS:
        subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]

        l_subs.append(subfolders)
    # Concatenation des listes de dossiers (list of lists) en un unique vecteur
    l_subs = [folder for folders in l_subs for folder in folders]

    response = ABSTRACT_CLASS in l_subs

    if response:
        print('Entrain de réaliser une restructuration des données brutes. Attendez svp.\n')
    else:
        print('Les données sont valides et peuvent être passées en entrée au modèle.\n')

    return response


#-------------------------------------------------------------------------------#
# Obtenir la liste des classes pour chaque ensemble
def classes_for_each_set(l_DIRS):
    '''
        -- l_DIRS : liste des répertoires des train, test et val sets du project.
    '''

    d = {} 
    for folder in l_DIRS:
        subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]

        if 'train'.upper() in folder.upper():
            d['TRAIN'] = subfolders

        elif 'test'.upper() in folder.upper():
            d['TEST'] = subfolders

        else:
            d['VALIDATION'] = subfolders

    return d


#-------------------------------------------------------------------------------#
# Original data structuring for specific schema
def structure_origin_data(l_DIRS, IMAGE_FORMAT, POSITIVE_CLASS):
    '''
        -- l_DIRS : liste des répertoires des train, test et val sets du project.
        -- IMAGE_FORMAT : format des fichiers à lire.
        -- POSITIVE_CLASS : nom de la classe "Positive" pour la tache.
    '''

    cwd = os.getcwd() 
    print('Répertoire de travail courant: {}'.format(cwd))

    dims_w = []
    dims_h = []

    dirs_to_delete = []

    for this_dir in l_DIRS:
        print('\n*** Gestion de lenvironnement de travail: {} ***'.format(this_dir))
        # Sauvegarder le répertoire temporaire
        temp_dir = '{}/{}'.format(cwd, this_dir) 
        os.chdir(temp_dir)

        # creer la liste des dossiers dans le répertoire temporaire
        list_folders = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
        print('| Dossiers trouvés:\n|| {}'.format(list(list_folders)))
        
        if len(list_folders) > 0:

            # Allez dans les sous-dossiers
            for this_folder in list_folders:
                #1. Changer le CWD pour celui du sous-dossier
                os.chdir(this_folder)
                print('\n| Entering to: {}'.format(this_folder))
                
                classes = [] # Classes dans ce sous-dossier
                files = []   # Liste de fichiers dans ce sous-dossier 
                
                #2. Récupération des images
                for image_file in glob.glob('*.{}'.format(IMAGE_FORMAT)):
                    
                    d = {} # Données temporaires pour une unique Image dans le sous-dossier
                    
                    #3. Ouvrir l'image pour extraire la donnée
                    with Image.open(image_file) as opened_image:    # Ouvrir le fichier image
                        opened_image_dims = opened_image.size       # Extraire les dimmensions
                        opened_image.close()                        # Fermer le fichier image
                    
                    #4. Identifier la classe (label)
                    if '_' in image_file:
                        this_class = image_file.upper().split('_')[1].split('_')[0].replace(' ', '')
                    else:
                        this_class = POSITIVE_CLASS

                    classes.append(this_class)
                    
                    #5. Assigner le fichier à la classe
                    d['Class'] = this_class
                    d['Filename'] = image_file
                    files.append(dict(d))
                    
                    #6. Ajouter les dimensions à la listes
                    dims_w.append(opened_image_dims[0])
                    dims_h.append(opened_image_dims[1])
                    
                #7. Obtenir les classes (uniques) dans un sous-dossier 
                unique_classes = list(set(classes))
                files_df = pd.DataFrame(files)

                print('||| Unique classes trouvées: {}'.format(unique_classes))
                print('||| Fichier trouvés au total: {}'.format(len(files)))

                for this_class, i in zip(unique_classes, range(0, len(unique_classes))):

                    print('--> {} = {}'.format(this_class, len(files_df[files_df['Class'] == this_class])))
                    dir_for_class = '{}/{}'.format(temp_dir, this_class)

                    #8. Générer la liste de fichiers à bouger
                    filelist = list(files_df[files_df['Class']==this_class]['Filename'])
                    print('|||--> La liste de fichier pour {} est créee.'.format(this_class))

                    # Faire une copie dans un dossier parent
                    #9. Crée un dossier spécial pour une classe spécifique dans un dossier parent
                    os.chdir(temp_dir)
                    
                    if len(unique_classes) > 1:
                        try:  

                            os.mkdir(dir_for_class)
                            print('|||| Création réussie du repertoire pour {} dans {}'.format(this_class, dir_for_class))

                        except OSError:

                            print('!!!! La création du repertoire {} a échoué'.format(dir_for_class))

                        os.chdir(this_folder)
                        for f in filelist:
                            shutil.copy(f, dir_for_class)

                        #11. Ajouter un fichier courrant à la liste pour supression 
                        dirs_to_delete.append(this_folder)
                        
                    elif (len(unique_classes) == 1) and (unique_classes[0].upper() != POSITIVE_CLASS.upper()):
                        dir_for_class = '{}/{}'.format(temp_dir, unique_classes[0].upper())
                        rename_folder_with_single_class(this_folder, dir_for_class)
                                      
    #12. Suppression des répertoires utilisés
    time.sleep(5)
    os.chdir(cwd)
    for deleting in set(dirs_to_delete):

        print('\n--> Suppression du dossier: {}'.format(deleting))
        delete_files_in_folder(deleting)
            
    print('*** Tous les dossiers utilisés ont été retiré du système. ***')
                
    # Calcul des dimmensions moyennes des images
    avg_dims_w = int(np.average(dims_w))
    avg_dims_h = int(np.average(dims_h))
    print('\n: Average Image Width = {}'.format(avg_dims_w))
    print(': Average Image Height = {}'.format(avg_dims_h))

    return None


#-------------------------------------------------------------------------------#
# Plot des graphs pour l'Accuracy, la Loss et la Validation
def plot_model_result(model):
    '''
    -- modèle : Keras model.
    '''

    rcParams['figure.figsize'] = 14, 4 # Fixe la taille du plot

    # Plot #1

    y1 = model.history.history['val_accuracy']
    y2 = model.history.history['accuracy']

    _ = plt.title('Model Results', family='Arial', fontsize=12)

    _ = plt.plot(y1, 
    color='blue', linewidth=1.5, marker='D', markersize=5,
    label='Validation acc.')
    _ = plt.plot(y2, 
    color='#9999FF', linewidth=1.5, marker='D', markersize=5,
    label='Training acc.')

    _ = plt.xlabel('Epochs', family='Arial', fontsize=10)
    _ = plt.ylabel('Score', family='Arial', fontsize=10)

    _ = plt.yticks(np.arange(0., 1.25, 0.1),
       family='Arial', fontsize=10)

    if len(model.history.history['accuracy']) < 51:
        _ = plt.xticks(np.arange(0, len(model.history.history['accuracy']), 1),
                                   family='Arial', fontsize=10)

    _ = plt.ylim((0., 1.))

    _ = plt.fill_between(np.arange(0, len(model.history.history['accuracy']), 1),
     model.history.history['accuracy'], 0,
     color = '#cccccc', alpha=0.5)

    _ = plt.grid(which='major', color='#cccccc', linewidth=0.5)
    _ = plt.legend(loc='best', shadow=True)
    _ = plt.margins(0.02)

    _ = plt.show()

    # Plot #2
    _ = plt.clf()

    _ = plt.plot(model.history.history['val_loss'], 
    color='red', linewidth=1.5, marker='D', markersize=5,
    label='Validation loss')
    _ = plt.plot(model.history.history['loss'], 
    color='#FF7F7F', linewidth=1.5, marker='D', markersize=5,
    label='Loss')

    _ = plt.xlabel('Epochs', family='Arial', fontsize=10)
    _ = plt.ylabel('Loss score', family='Arial', fontsize=10)

    if len(model.history.history['accuracy']) < 51:
        _ = plt.xticks(np.arange(0, len(model.history.history['accuracy']), 1),
                                   family='Arial', fontsize=10)
    _ = plt.yticks(family='Arial', fontsize=10)

    _ = plt.grid(which='major', color='#cccccc', linewidth=0.5)
    _ = plt.legend(loc='best', shadow=True)
    _ = plt.margins(0.02)

    _ = plt.show()

    return None


#-------------------------------------------------------------------------------#
# Convertion des images RGB en images en noir et blanc 
def rgb_to_grayscale(imgs_set):
    '''
    -- imgs_set : set of RGB images (3 channels)
    '''

    return tf.image.rgb_to_grayscale(imgs_set, name=None)


#-------------------------------------------------------------------------------#
# Sauvegarder les résultats du modèle dans un DataFrame, puis dans un fichier CSV
def save_model_result(model):
    '''
    -- model : compiled model on the given data.
    '''

    # Extraction des résultats du modèle
    data_val_acc = list(model.history.history['val_accuracy'])
    data_acc = list(model.history.history['accuracy'])
    data_val_loss = list(model.history.history['val_loss'])
    data_loss = list(model.history.history['loss'])

    # Sauvegarde les résultats du modèle dans un dictionnaire, puis un DataFrame 
    d = {}
    d['val_acc'] = data_val_acc
    d['acc'] = data_acc
    d['val_loss'] = data_val_loss
    d['loss'] = data_loss

    df = pd.DataFrame(d)
    print(df)

    # Obtenir la "timestamp" courante
    timestamp = str(datetime.datetime.now()).replace(":","-")[:-10].replace(' ', '_')
    filename = 'model_results_{}.csv'.format(timestamp)
    df.to_csv(filename, encoding='utf-8')

    print('\n\nLes résultat du modèle sont sauvegardés dans le fichier: {}'.format(filename))

    return df