import pandas as pd
from feature_selector import FeatureSelector

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


RAT_trojans = ["NingaliNETServer.exe", "RabbitHoleServer.exe", "RevengeRATServer.exe"]

def ForEvents(file_csv):
    data = pd.read_csv(file_csv)
    # data = data.drop(["Events_Charact.Time_Stamp_Start"], axis=1)
    # data = data.drop(["Events_Charact.Time_Stamp_End"], axis=1)
    # data = data.drop(["Events_Charact.Process_Name"], axis=1)
    # data = data.drop(["Events_Charact.Direction_IP_Port"], axis=1)
    # data = data.drop(["Events_Charact.Count_Events_Batch"], axis=1)
    # data = data.drop(["Events_Charact.Duration"], axis=1)

    data = data.drop(["Flow_Charact.Time_Stamp_Start"], axis=1)
    data = data.drop(["Flow_Charact.Time_Stamp_End"], axis=1)
    data = data.drop(["Flow_Charact.Src_IP_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Dst_IP_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Src_Port_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Dst_Port_Flow"], axis=1)

    print(data)

    metki = {"metka": []}

    Src_IP_Flow = list(pd.read_csv(file_csv).to_dict("series")["Flow_Charact.Src_IP_Flow"])
    Dst_IP_Flow = list(pd.read_csv(file_csv).to_dict("series")["Flow_Charact.Dst_IP_Flow"])
    for i in range(len(Src_IP_Flow)):
        if Src_IP_Flow[i] == 3232238209 or Dst_IP_Flow[i] == 3232238209:
            metki["metka"].append(1)
        else:
            metki["metka"].append(0)

    fs = FeatureSelector(data=data, labels=pd.DataFrame(metki))

    print("1. Отсутствующие значения")
    fs.identify_missing(missing_threshold=0.0)
    missing_features = fs.ops['missing']
    print(missing_features)
    print(fs.missing_stats.head(150))
    for st in fs.missing_stats.head(150):
        print(st)

    print("\n\n2. Одно уникальное значение")
    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    for su in single_unique:
        print(su)
    fs.unique_stats.sample(59).to_csv("F:\\DataSets\\TRAFFIC\\Выбор характеристик\\2. Одно уникальное значение.csv")
    print(fs.unique_stats.sample(59))

    print("\n\n3. Коллинеарные (сильно коррелированные) признаки")
    fs.identify_collinear(correlation_threshold=0.9)
    fs.plot_collinear()
    fs.plot_collinear(plot_all=True)
    correlated_features = fs.ops['collinear']
    for cf in correlated_features:
        print(cf)
    print(f"Всего признаков выбрано: {len(correlated_features)}")

    print("\n\n4. Признаки нулевой важности")
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                                n_iterations=100, early_stopping=True)

    one_hot_features    = fs.one_hot_features
    base_features       = fs.base_features
    print('Количество оригинальных функций: %d' % len(base_features))
    print('Количество горячих функций: %d' % len(one_hot_features))
    fs.data_all.head().to_csv("F:\\DataSets\\TRAFFIC\\Выбор характеристик\\4. Признаки нулевой важности.csv")
    print(fs.data_all.head())

    zero_importance_features = fs.ops['zero_importance']
    for zif in zero_importance_features:
        print(zif)
    print(f"Всего признаков выбрано: {len(zero_importance_features)}")

    print("\n\n5. Признаки низкой важности")
    fs.identify_low_importance(cumulative_importance=0.99)
    low_importance_features = fs.ops['low_importance']
    for lif in low_importance_features:
        print(lif)
    print(f"Всего признаков выбрано: {len(low_importance_features)}")


if __name__ == '__main__':
    train_file_events = "F:\\DataSets\\EVENT\\train_dataset_VictimPC\\train_dataset_VictimPC.csv"
    test_file_events = "F:\\DataSets\\EVENT\\test_dataset_VictimPC\\test_dataset_VictimPC.csv"

    for_choose = "F:\\DataSets\\Для выбора характеристик\\Все в одном.csv"
    traffic = "F:\\DataSets\\TRAFFIC\\Выбор характеристик\\Для_анализа_характеристик.csv"

    feature_events = ForEvents(traffic)
    # print(feature_events)

