# IA-Projet-Deployement-de-Modele-Flask-GCP
Présente un modele d'IA (classification) déployé en production avec Flask et Google Cloud Platform (GCP)

Ce premier projet synthétise le savoir faire acquis jusqu'ici dans le domaine de la Vision par Ordinateur (Computer Vision ou CV).
Il est le fruit de mes habitudes de code acquises lors de mes expériences professionnelles (stages et alternance) mais aussi et surtout 
une application des notions vues lors de moocs sur le "Computer Vision" que j'ai suivi jusque là (Udemy et FastAi principalement). 

Dans ce projet sont inclus les aspects Back-End (Keras et Flask) et Front-End (HTML, CSS, JS) du travail de déploiement.
On va entrainer un modèle de classification basé sur les Convolutional Neural Networks (CNN) afin de classifier les images issues de radiographies d'un poumon.
Les trois états de classification sont : Bacterie, Virus et Normal. Le but étant de diagnostiquer une pneumonie chez un patient avec une bonne précision.

La précision du modèle est approximatetivement de 80% (avec l'aide de technique commme le Dropout, la Régularisation L2 et l'optimisateur Adam)

Pour le déploiement du modèle on utilise Flask pour sa simplicité de prise en main et sa compatibilité avec Google Cloud Platform (GCP).
On utilise également App Engine (service de la suite GCP) pour déployer notre modèle Keras comme une application web.
Enfion on crée un fichier Python YAML pourque le serveur comprenne quelle est la configuration de notre application.


Bonus : Utilisation de HTML, CSS and Javascript pour "pimper" le modèle front-end. Simplicité et facile d'utilisation, même compatible avec les téléphones mobiles !

Lien vers l'application web :  https://flaskdeploy-16092021.appspot.com 

Lien vers mon profil LinkedIn : https://www.linkedin.com/in/ralph-tankoua-06066715b/
