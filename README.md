# OC_IML_P7 - Développez une preuve de concept

**Description du projet**:

Le domaine du machine learning, et plus généralement de la data science évolue très rapidement. Il est donc important de réaliser régulièrement 
de la veille technologique pour tester l'efficacité des nouvelles méthodes, à l'aide de 'proof-of-concepts' (POCs)

La méthode étudiée ici est le résau de neurone ConvNeXtTiny.
Modèle de réseau de neurones inspiré des Transformers.

L'objectif est d'implémenter cette méthode et de tester ses performances sur de la classification d'images et notamment 
sa robustesse à l'augmentation du nombre de classes.

Nous travaillons avec la biblitothèque Keras, pour le modèle de base Resnet50, et également pour le modèle à tester : ConvNeXtTiny.

**Ressources** :
* [Repo ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
* [Bibliothèque Keras](https://keras.io/api/applications/convnext/)

**Données**:
* [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/): Dataset issu d'ImageNet comportant 20 580 images réparties en 120 classes de chiens.

**Livrables**:
* [Plan de travail prévisionnel](Plan_prévisionnel.docx): plan de travail prévisionnel
* [Notebook de préparation](IML_P7_V5-5races.ipynb): préparation de données et d’entraînement des algorithmes (baseline + nouvelle méthode) et le jeu de données.
* [Notebook de création de l'app](IML_P7_ModelV3.ipynb): génération de l'app.py
* [Note méthodologique](Note_Méthodologique.docx): note méthodologique présentant la preuve de concept
* [code du dashboard](app.py): app.py à déployer dans streamlit
* [application cloud](https://ocimlp7-jj.streamlit.app/): lien vers streamlit
* [support de présentation](IML_P7_JJ_V1.pdf): présentation ppt