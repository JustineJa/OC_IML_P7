# OC_IML_P7

**Description du projet**:

Le domaine du machine learning, et plus généralement de la data science évolue très rapidement. Il est donc important de réaliser régulièrement 
de la veille technologique pour tester l'efficacité des nouvelles méthodes, à l'aide de 'proof-of-concepts' (POCs)

La méthode étudiée ici est le résau de neurone ConvNeXtTiny.
Modèle de réseau de neurones inspiré des Transformers.

L'objectif est d'implémenter cette méthode et de tester ses performances sur de la classification d'images et noamment 
à l'augmentation du nombre de classes.

Nous travaillons avec la biblitothèque Keras, à la fois por le modèlde de base Resnet50, et également pour le modèle à tester : ConvNeXtTiny.
[https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

**Données**:
* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/): Dataset issu d'ImageNt comportant 20 580 images réparties en 120 classes de chiens.

**Livrables**:
* [Plan de travail prévisionnel](7_layers_cifar.ipynb): plan de travail prévisionnel
* [Notebook de préparation](7_layers_dogs.ipynb): préparation de données et d’entraînement des algorithmes (baseline + nouvelle méthode) et le jeu de données.
* [Notebook de création de l'app](7_layers_dogs.ipynb): génération de l'app.py
* [Note méthodologique](9_layers_cifar.ipynb): note méthodologique présentant la preuve de concept
* [code du dashboard](9_layers_dogs.ipynb): app.py à déployer dans streamlit
* [application cloud](https://ocimlp7-jj.streamlit.app/): lien vers streamlit
* [support de présentation](xxx): présentation ppt


