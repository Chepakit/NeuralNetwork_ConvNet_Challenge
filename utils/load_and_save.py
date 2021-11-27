'''
Ingénierie des modèles — ConvNet challenge

Léonard Benedetti, 2020
------------------------------------------

L’objectif de ce fichier est de vous montrer comment faire pour
sauvegarder un modèle Keras pour pouvoir le réutiliser ; et comment
faire pour charger un modèle sauvegardé pour pouvoir l’exploiter.

Notez qu’il s’agit d’extraits qu’il convient d’adapter à votre
situation.
'''

# ---------------------
# Sauvegarder un modèle
# ---------------------

# On suppose que le modèle à sauvegarder est dans la variable `model`

model.save('./my_model.keras')


# --------------------------------
# Charger un modèle et l’exploiter
# --------------------------------


from tensorflow.keras.models import load_model

model = keras.models.load_model('./my_model.keras')

# Pour exploiter le modèle, n’oubliez pas d’appliquer les *mêmes* opérations de preprocessing (normalisation identique, etc.), sur les données qui lui sont données en entrée, que pendant l’entraînement !
