"""
    Модуль для обучения нейронной сети распознавать вредоностный трафик RAT-троянов
"""

import pickle
import numpy as np
import pandas as ps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess(data, seed=None):
    '''
        Осуществить предварительную обработку данных,
        масштабирование и маркировку.
        Аргументы:
            data - DataFrame таблицы признаков
            seed - семя генератора псевдослучайных чисел
        Возвращает:
            DataFrame с обработанной таблицей признаков,
            StandardScaler,
            LabelEncoder
    '''

    # Перемешиваем данные:
    if seed:
        np.random.seed(seed)
        data = data.iloc[np.random.permutation(len(data))]
    print("Массив данных успешно перемешан!")

    # Масштабируем и маркируем:
    scaler = StandardScaler()
    labeler = LabelEncoder()
    X = scaler.fit_transform(data.drop(["metka"], axis=1))
    y = labeler.fit_transform(data["metka"])

    cols = [col for col in data.columns if col not in "metka"]
    print("Масштабирование и маркировака успешно произведены!")
    return ps.concat([ps.DataFrame(X, columns=cols),
                      ps.DataFrame({"metka": y})], axis=1), scaler, labeler


def split_data(data):
    '''
        Разделить равномерно таблицу признаков на три части
        Аргументы:
            data - DataFrame таблицы признаков
        Возвращает:
            Список из трёх DataFrame, каждый
            размером 1/10 от оригинала
    '''
    array_data = [data[data["metka"] == metka] for metka in data["metka"].unique()]

    clusters = [[], []]
    for cluster in array_data:
        split_index = len(cluster) // 10
        clusters[0].append(cluster.iloc[0:split_index * 9])
        clusters[1].append(
                cluster.iloc[split_index * 9:])

    arr = [ps.concat(clus) for clus in clusters]
    return arr[0], arr[1]


def train_model(data_train, seed=None):
    '''
        Обучить модель на таблице признаков.
        Аргументы:
            data_train - DataFrame обучающей выборки
            seed - семя генератора псевдослучайных чисел
        Возвращает:
            Обученную модель RandomForest
    '''
    X_train = data_train.drop(["metka"], axis=1)
    y_train = data_train["metka"]
    model = RandomForestClassifier(27, criterion="entropy", max_depth=9, random_state=seed)
    model.fit(X_train.values, y_train)
    print("Обучение успешно завершено!")
    return model


def score_model(model, data_test, labeler):
    '''
        Оценить производительность модели,
        выведя в стандартный вывод три таблицы:
        важности признаков, значения полноты и
        точности для каждого класса, реальные
        и предсказанные классы.
        Аргументы:
            model - обученная модель
            data_test - проверочаня выборка
            labeler - LabelEncoder данной выборки
        Возвращает:
            Ничего
    '''
    X_test = data_test.drop(["metka"], axis=1)
    y_test = data_test["metka"]
    y_predicted = model.predict(X_test)

    true_labels = labeler.inverse_transform(y_test)
    predicted_labels = labeler.inverse_transform(y_predicted)

    print(feature_importances_report(model, X_test.columns))
    print("\n", classification_report(true_labels, predicted_labels))
    print(cross_class_report(true_labels, predicted_labels))


def cross_class_report(y, p):
    '''
        Составить таблицу реальных и предсказанных
        классов.
        Аргументы:
            y - numpy-массив реальных меток классов
            p - numpy-массив предсказанных меток классов
        Возвращает:
            DataFrame
    '''
    classes = np.unique(y)
    res = ps.DataFrame({"y": y, "p": p}, index=None)
    table = ps.DataFrame(index=classes, columns=classes)
    for true_cls in classes:
        tmp = res[res["y"] == true_cls]
        for pred_cls in classes:
            table[pred_cls][true_cls] = len(tmp[tmp["p"] == pred_cls])
    return table


def feature_importances_report(model, columns):
    '''
        Составить отчёт о важности признаков.
        Аргументы:
            model - обученная модель RandomForest
            columns - список текстовых наименований
                признаков
        Возвращает:
            Отчёт в виде строки
    '''
    imp = {col: imp for imp, col
           in zip(model.feature_importances_, columns)}
    assert len(imp) == len(columns)
    return "\n".join("{}{:.4f}".format(str(col).ljust(25), imp)
                     for col, imp in sorted(imp.items(), key=(lambda x: -x[1])))


def main():
    data = ps.read_csv("../data/to_learn.csv")
    seed = 89459876

    data, scaler, labeler = preprocess(data, seed)
    data_learn, data_test = split_data(data)

    model = train_model(data, seed)
    score_model(model, data_test, labeler)


    pickle.dump((model, scaler, labeler), open("../mdl/madel_traffic.mdl", "wb"))
    print("\nМодель успешно записана в файл '{}'.".format("madel_traffic.mdl"))

if __name__ == "__main__":
    main()
