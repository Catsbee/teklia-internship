# library importation
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib


# take into account arguments when launching the script
arg = sys.argv
if len(arg) < 2:
    print("Wrong number of arguments")
else:
    # Script for the train command
    if arg[1] == "train":
        #1 for script_python_command.py, 1 for test, 1 for test set, 1 for saving path
        if len(arg) != 4:
            print("Unexpected number of argument for train")
        else:
            training_set = arg[2]
            saving_path = arg[3]
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
    elif arg[1] == "test":
        #1 for script_python_command.py, 1 for for train,
        if len(arg) != 5:
            print("Unexpected number of argument for test")
        else:
            testing_set = arg[2]
            model = arg[3]
            pred = arg[4]
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
    else:
        print("Command not recognised")

print(arg)