from AutoEncoder_RNN import *
from pathlib import Path

def main(versia, arhiteche):
    # Параметры датасета
    batch_size          = 1000
    validation_factor   = 0.05
    window_size         = 1000
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
    shuffle             = False
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\TrafficAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_TAD_v" + versia
    max_min_file        = path_model + "M&M_traffic_VNAT.csv"
    dataset             = "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_all.csv"
    history_name        = path_model + "history_train_v" + versia + ".csv"
    history_valid_name  = path_model + "history_valid_v" + versia + ".csv"

    if not Path(path_model).exists():
        Path(path_model).mkdir()

    if not Path(path_model + "Checkpoint\\").exists():
        Path(path_model + "Checkpoint\\").mkdir()

    data = pd.read_csv(dataset)
    data = data.drop(["Time_Stamp"], axis=1)
    data = data.drop(["Count_src_is_dst_ports"], axis=1)
    data = data.drop(["Dev_size_TCP_paket"], axis=1)
    data = data.drop(["Dev_size_UDP_paket"], axis=1)
    data = data.drop(["Dev_client_paket_size"], axis=1)
    data = data.drop(["Dev_server_paket_size"], axis=1)

    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, feature_range, batch_size, window_size,
                                          validation_factor)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.caracts_count, arhiteche, window_size)
    autoencoder.build((1, window_size, training_dataset.caracts_count))
    autoencoder.summary()

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     init_learning_rate,
    #     decay_steps=decay_steps,
    #     decay_rate=decay_rate,
    #     staircase=staircase
    # )
    #
    # optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    autoencoder.compile(optimizer="adam", loss=loss_func)
    print("Автоэнкодер определён.")

    if continue_education:
        checkpoint_name = "modeles\\TrafficAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_" + str(checkpoint)
        autoencoder.load_weights(checkpoint_name)
        print("Продолжаем обучение:")
    else:
        checkpoint = None
        print("Начинаем обучение:")

    autoencoder.education(training_dataset, epochs=epochs, shuffle=shuffle,
                          model_checkname=path_model + "Checkpoint\\", versia=versia,
                          path_model=path_model, checkpoint=checkpoint)
    autoencoder.save(model_name)

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)


if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    print("Ожидаем начала обучения!")
    # time.sleep(9000)
    print("Запускаем обучение!")

    arhiteche = {"GRU_1": (14, 15), "GRU_2": (13, 14), "GRU_3": (12, 13), "GRU_4": (11, 12),
                 "GRU_5": (12, 11), "GRU_6": (13, 12), "GRU_7": (14, 13), "GRU_8": (15, 14)}
    versia = "0.8.7.2"
    print("\n\n" + versia)
    main(versia, arhiteche)