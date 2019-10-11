"""Clasifica el conjunto de entrada usando el mejor clasificador encontrado"""

import sys
sys.path.append("notebooks/")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentiment import PCA, KNNClassifier

def get_instances(df, df_test):
    """
    Lee instancias de entrenamiento y de test
    """
    trainData = df[df.type == 'train'].sample(n=4000, random_state=123)

    text_train = trainData["review"]
    label_train = trainData["label"]

    text_test = df_test["review"]
    ids_test = df_test["id"]

    print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
    print("Cantidad de instancias de test = {}".format(len(text_test)))
    vectorizer = CountVectorizer(max_df=0.95, min_df=0.0,max_features=5000)

    vectorizer.fit(text_train)

    X_train, y_train = vectorizer.transform(text_train).toarray(), (label_train == 'pos').values

    X_test = vectorizer.transform(text_test).toarray()

    return X_train, y_train, X_test, ids_test

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python classify archivo_de_test archivo_salida")
        exit()

    test_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv("data/imdb_small.csv")
    df_test = pd.read_csv(test_path)

    print("Vectorizando datos...")
    X_train, y_train, X_test, ids_test = get_instances(df, df_test)

    """
    Entrenamos KNN
    """
    clf = KNNClassifier(1120)

    clf.fit(X_train, y_train)

    """
    Testeamos
    """
    print("Prediciendo etiquetas...")
    y_pred = clf.predict(X_test).reshape(-1)

    labels = ['pos' if val == 1 else 'neg' for val in y_pred]

    df_out = pd.DataFrame({"id": ids_test, "label": labels})

    df_out.to_csv(out_path, index=False)

    print("Salida guardada en {}".format(out_path))
