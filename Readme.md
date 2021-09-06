# Comment executer le projet

## Création du script d'entrainement du modèle
```bash
python parser.py
```
Ce script créer un fichier output.py que j'ai renommé `script_python_command.py` puis que j'ai modifié pour qu'il marche en ligne de commande avec un choix entre le train et le test

## Utilisation du script d'entrainement
Pour lancer l'entrainement du modèle, lancer la commande suivante :
```bash
python script_python_command.py train LeMonde2003_9classes.csv sav
```

Pour tester le modèle, lancer la commande suivante :
```bash
python script_python_command.py test LeMonde2003_9classes.csv sav t_sav
```