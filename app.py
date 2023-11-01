import random

from flask import request
from blocklibs.chain.blockchain import Blockchain
from flask import Flask
import datetime
import json
import requests
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
bc = Blockchain()


@app.route('/new_transaction', methods=['GET', 'POST'])
def new_transaction():
    print()
    incoming_transaction = request.values.to_dict()
    required_data = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    for mandatory_field in required_data:
        if not incoming_transaction.get(mandatory_field):
            return "Incoming transaction is invalid", 404

    incoming_transaction["timestamp"] = str(datetime.datetime.now())
    bc.add_new_transaction(incoming_transaction)

    return "Transaction added, pending to validate", 201


@app.route('/add_data', methods=['GET'])
def add_data():
    import pandas as pd

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = pd.read_csv(url, names=names)

    for i, j in dataset.iterrows():
        requests.post(
            f"http://127.0.0.1:8000/new_transaction?sepal-length={j['sepal-length']}&sepal-width={j['sepal-width']}&petal-length={j['petal-length']}&petal-width={j['petal-width']}&Class={j['Class']}")
        requests.get("http://127.0.0.1:8000/mine")

    return "Data added", 201

# equivalent to chain
@app.route('/node', methods=['GET'])
def get_node():
    node_data = []
    for block in bc.chain:
        node_data.append(block.get_block)
    return json.dumps({"length": len(node_data),
                       "chain": node_data})


@app.route('/mine', methods=['GET'])
def mine_unconfirmed_transactions():
    result = bc.compute_transactions()
    if not bc:
        return "not transactions to mine"
    return "Block #{} mined".format(result)


@app.route('/check')
def check():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = read_csv(url, names=names)
    if bc.check_data_integrity(dataset):
        return "Data integrity successfully checked."
    return "Data integrity check failed"


@app.route('/pending_tx')
def get_pending_tx():
    return json.dumps(bc.unconfirmed_transactions)


@app.route('/study')
def study(sklearn=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = read_csv(url, names=names)

    dataset.loc[(dataset['sepal-length'] > 0), 'sepal-length'] = random.random()
    dataset.loc[(dataset['sepal-width'] > 0), 'sepal-width'] = random.random()
    dataset.loc[(dataset['petal-length'] > 0), 'petal-length'] = random.random()
    dataset.loc[(dataset['petal-width'] > 0), 'petal-width'] = random.random()

    if len(bc.chain) == 1:
        add_data()
    else:
        if not bc.check_data_integrity(dataset):
            return "Data integrity check failed!"

    # Разделение датасета на обучающую и контрольную выборки
    array = dataset.values

    # Выбор первых 4-х столбцов
    X = array[:, 0:4]

    # Выбор 5-го столбца
    y = array[:, 4]

    # Разделение X и y на обучающую и контрольную выборки
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Загружаем алгоритмы модели
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    # оцениваем модель на каждой итерации
    results = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Сравниванием алгоритмы
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()

    # Создаем прогноз на контрольной выборке
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Оцениваем прогноз
    print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    print(dataset)

    return "Check IDLE"


if __name__ == '__main__':
    app.run(debug=True, port=8000)


