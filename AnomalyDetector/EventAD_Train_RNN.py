import keras.losses

from AutoEncoder_RNN import *
from pathlib import Path


def main(versia, window_size, arhiteche):
    # Параметры датасета
    batch_size          = 1000
    validation_factor   = 0.05
    feature_range       = (-1, 1)

    # Параметры оптимизатора
    init_learning_rate  = 0.001
    decay_steps         = 50000
    decay_rate          = 0.1
    staircase           = True

    # Параметры нейронной сети
    epochs              = 3
    continue_education  = True
    checkpoint          = None
    checkpoint_epoch    = 0
    sdvig               = True
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\EventAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_EAD_v" + versia
    max_min_file        = path_model + "M&M_event.csv"
    dataset             = "F:\\EVENT\\EventTest\\train_112_dataset_2m_1.csv"
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

    data = pd.read_csv(dataset)
    data = data.drop(["Time_Stamp_Start"], axis=1)
    data = data.drop(["Time_Stamp_End"], axis=1)
    data = data.drop(["Process_Name"], axis=1)
    data = data.drop(["Count_Events_Batch"], axis=1)
    data = data.drop(["Count_System_Statistics"], axis=1)
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, feature_range, checkpoint_epoch,
                                          batch_size, window_size, validation_factor)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.caracts_count, arhiteche, window_size)
    autoencoder.build((batch_size, window_size, training_dataset.caracts_count))
    autoencoder.summary()
    autoencoder.graph.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        init_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    print("Автоэнкодер определён.")

    checkpoint_name = "modeles\\EventAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_1"
    autoencoder.load_weights(checkpoint_name)
    autoencoder.save(model_name+"batch_1000")
    exit(-1)

    if continue_education:
        checkpoint_name = "modeles\\EventAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_1" + checkpoint
        autoencoder.load_weights(checkpoint_name)
        print(f"Продолжаем обучение с контрольной точки: {checkpoint}")
    else:
        checkpoint = None
        print("Начинаем обучение:")

    autoencoder.education(training_dataset, epochs=epochs, sdvig=sdvig,
                          model_checkname=path_model + "Checkpoint\\", versia=versia,
                          path_model=path_model, checkpoint=checkpoint_epoch)
    autoencoder.save(model_name)

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)


if __name__ == '__main__':
    versia = "0.4.7.1_LSTM"
    window_size = 1
    arhiteche = {"1_Input": (window_size, 112),
                 "2_LSTM_seq": (56, 112), "3_LSTM_seq": (28, 56), "4_GRU": (14, 28),
                 "5_RepeatVector": (window_size, None),
                 "6_GRU_seq": (28, 14), "7_LSTM_seq": (56, 28), "8_LSTM": (112, 56)}

    print("\n\n" + versia)
    main(versia, window_size, arhiteche)