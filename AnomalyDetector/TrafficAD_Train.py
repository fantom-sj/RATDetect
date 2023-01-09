from AutoEncoder_RNN import *
from pathlib import Path
from sklearn.ensemble import IsolationForest
import pickle

def main(versia, arhiteche, window_size):
    # Параметры датасета
    batch_size          = 1
    validation_factor   = 0.05
    feature_range       = (-1, 1)

    # Параметры оптимизатора
    init_learning_rate  = 0.1
    decay_steps         = 1500
    decay_rate          = 0.96
    staircase           = True

    # Параметры нейронной сети
    epochs              = 1
    continue_education  = False
    checkpoint          = None
    checkpoint_epoch    = 0
    shaffle             = False
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\TrafficAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_TAD_v" + versia
    max_min_file        = path_model + "M&M_traffic_VNAT.csv"
    dataset             = "F:\\VNAT\\Mytraffic\\"
    history_name        = path_model + "history_train_v" + versia + ".csv"
    history_valid_name  = path_model + "history_valid_v" + versia + ".csv"

    if continue_education:
        if Path(path_model+"\\Checkpoint\\checkpoint").exists():
            with open(path_model+"\\Checkpoint\\checkpoint", "r") as file:
                str1 = file.readline()
                idx = str1.find(': "') + 3
                checkpoint = str1[idx:-2]
                checkpoint_epoch = int(checkpoint[6:])
                print(checkpoint_epoch)
        else:
            continue_education = False

    if not Path(path_model).exists():
        Path(path_model).mkdir()

    if not Path(path_model + "Checkpoint\\").exists():
        Path(path_model + "Checkpoint\\").mkdir()

    data = pd.read_csv(dataset + "dataset_all.csv")
    # data = pd.concat([data, pd.read_csv(dataset + "VNAT_novpn_youtube.csv")], ignore_index=True) InSDN_dataset_0 47300
    # data = pd.concat([data, pd.read_csv(dataset + "VNAT_vpn_youtube.csv")], ignore_index=True)
    data = data[:3475]
    print(f"Загружено {len(data)} характеристик")

    data = data.drop(["Flow_Charact.Time_Stamp_Start"], axis=1)
    data = data.drop(["Flow_Charact.Time_Stamp_End"], axis=1)
    data = data.drop(["Flow_Charact.Src_IP_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Dst_IP_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Src_Port_Flow"], axis=1)
    data = data.drop(["Flow_Charact.Dst_Port_Flow"], axis=1)

    # print(data.to_dict("records"))
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, feature_range, 0,
                                          batch_size, window_size, validation_factor)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.caracts_count, arhiteche, window_size, batch_size)
    autoencoder.build((1, window_size, training_dataset.caracts_count))
    autoencoder.summary()
    autoencoder.encoder_model.summary()
    autoencoder.decoder_model.summary()

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     init_learning_rate,
    #     decay_steps=decay_steps,
    #     decay_rate=decay_rate,
    #     staircase=staircase
    # )
    #
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    print("Автоэнкодер определён.")

    if continue_education:
        checkpoint_name = "modeles\\TrafficAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_" + str(checkpoint)
        autoencoder.load_weights(checkpoint_name)
        print("Продолжаем обучение:")
    else:
        checkpoint = None
        print("Начинаем обучение:")

    autoencoder.education(training_dataset, epochs=epochs, shaffle=shaffle,
                          model_checkname=path_model + "Checkpoint\\", versia=versia,
                          path_model=path_model, checkpoint=checkpoint)
    autoencoder.save(model_name)

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)

    # anomalyDetector = IsolationForest()
    # noAnomaly = []
    # for idx in range(len(autoencoder.history_valid["epoch"])):
    #     if autoencoder.history_valid["epoch"][idx] == 2:
    #         noAnomaly.append(autoencoder.history_valid["loss"][idx])
    # anomalyDetector.fit(noAnomaly)
    #
    # with open("modeles\\TrafficAnomalyDetector\\" + versia + "\\anomalyDetector", "wb") as model_file:
    #     pickle.dump(anomalyDetector, model_file)


if __name__ == '__main__':
    versia = "0.9.7.3"

    # history = "D:\\Пользователи\\Admin\\Рабочий стол\\Статья по КБ\\" \
    #           "RATDetect\\AnomalyDetector\\modeles\\TrafficAnomalyDetector\\0.9.5\\history_valid_v0.9.5.csv"
    #
    # history = pd.read_csv(history)
    # anomalyDetector = IsolationForest()
    # noAnomaly = []
    # for idx in range(len(history["epoch"])):
    #     if history["epoch"][idx] == 2:
    #         noAnomaly.append(history["loss"][idx])
    # total = len(noAnomaly) - 20
    # bar = tqdm(total=total, desc="Обучение детектора аномалий")
    # for idx in range(0, total, 1):
    #     data = np.array(noAnomaly[idx:idx+20]).reshape(-1, 1)
    #     anomalyDetector.fit(data)
    #     bar.update(1)
    # bar.close()
    #
    # with open("modeles\\TrafficAnomalyDetector\\" + versia + "\\anomalyDetector", "wb") as model_file:
    #     pickle.dump(anomalyDetector, model_file)
    #
    # exit(-1)


    window_size = 5

    # encoder = {"1_Input": (window_size, 59), "2_GRU_seq": (50, 59),
    #            "3_GRU_seq": (40, 50), "4_GRU_seq": (30, 40), "5_GRU": (20, 30)}
    # decoder = {"6_RepeatVector": (window_size, None), "7_GRU_seq": (30, 20),
    #            "8_GRU_seq": (40, 30), "9_GRU_seq": (50, 40), "10_GRU": (59, 50)}

    encoder = {"1_Input": (window_size, 59), "2_GRU_seq": (45, 59),
               "4_GRU_seq": (30, 45), "5_GRU": (20, 30)}
    decoder = {"6_RepeatVector": (window_size, None), "7_GRU_seq": (30, 20),
               "9_GRU_seq": (45, 30), "10_GRU": (59, 45)}

    arhiteche = (encoder, decoder)
    print("\n\n" + versia)
    main(versia, arhiteche, window_size)