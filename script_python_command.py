# library importation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import argparse
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('--command', type=str, required=True, choices=["train","test"], help="either train or test")
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--model', type=str, required=False)

args = parser.parse_args()

# take into account arguments when launching the script


# Script for the train command
if args.command == 'train':
    #1 for script_python_command.py, 1 for test, 1 for test set, 1 for saving path


            training_set = args.data
            saving_path = args.output
            df_train = pd.read_csv(training_set)
            df_train_hyper, df_dev = train_test_split(df_train, test_size=0.25, random_state=42)

            X_train = df_train_hyper.text
            X_dev = df_dev.text
            y_train = df_train_hyper.category
            y_dev = df_dev.category

            X_train_dev = df_train.text
            y_train_dev = df_train.category

            # train a Naive Bayes classifier

            # create the vectorizer object
            vectorizer = CountVectorizer(max_features=10000)
            # fit on train data
            vectorizer.fit(X_train)
            # apply it on train and dev data
            X_train_counts = vectorizer.transform(X_train)
            X_dev_counts = vectorizer.transform(X_dev)

            tf_transformer = TfidfTransformer().fit(X_train_counts)
            X_train_tf = tf_transformer.transform(X_train_counts)
            X_dev_tf = tf_transformer.transform(X_dev_counts)

            clf2 = MultinomialNB()
            clf2.fit(X_train_tf, y_train)

            print("Train accuracy", clf2.score(X_train_counts, y_train))
            print("Dev accuracy", clf2.score(X_dev_counts, y_dev))

            export = {"model" : clf2, "vectorizer" : vectorizer, "tf_trans" : tf_transformer}
            #Exportation
            joblib.dump(export, saving_path)

            #For testing in command :
            #python script_python_command.py train LeMonde2003_9classes.csv sav
elif args.command == "test":
        #1 for script_python_command.py, 1 for for train,
          if args.model == None :
            print('Model argument mendatory for test command')
          else:
            testing_set = args.data
            model = args.model
            pred = args.output
            df_test = pd.read_csv(testing_set)

            X_test = df_test.text
            y_test = df_test.category

            loaded_obj = joblib.load(model)
            vectorizer = loaded_obj["vectorizer"]
            tf_transformer = loaded_obj["tf_trans"]
            clf2 = loaded_obj["model"]
            # fit on train data
            vectorizer.fit(X_test)

            X_test_counts = vectorizer.transform(X_test)

            X_dev_tf = tf_transformer.transform(X_test_counts)

            print("Test accuracy", clf2.score(X_test_counts, y_test))

            # Exportation
            f = open(pred, "w")
            f.writelines('\n'.join(clf2.predict(X_test_counts)))
            f.close()



