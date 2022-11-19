from AutoEncoder_RNN import *
from pathlib import Path


def main(versia, arhiteche):
    # Параметры датасета
    batch_size          = 100
    validation_factor   = 0.05
    window_size         = 100
    feature_range       = (-1, 1)

    # Параметры оптимизатора
    init_learning_rate  = 0.001
    decay_steps         = 10000
    decay_rate          = 0.25
    staircase           = True

    # Параметры нейронной сети
    epochs              = 3
    continue_education  = False
    checkpoint          = None
    shuffle             = False
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\EventAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_EAD_v" + versia
    max_min_file        = path_model + "M&M_event.csv"
    dataset             = "F:\\EVENT\\EventTest\\train_dataset_220.csv"
    history_name        = path_model + "history_train_v" + versia + ".csv"
    history_valid_name  = path_model + "history_valid_v" + versia + ".csv"

    if not Path(path_model).exists():
        Path(path_model).mkdir()

    if not Path(path_model + "Checkpoint\\").exists():
        Path(path_model + "Checkpoint\\").mkdir()

    data = pd.read_csv(dataset)
    data = data.drop(["Time_Stamp"], axis=1)
    data = data.drop(["Process_name"], axis=1)

    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, feature_range, batch_size, window_size,
                                          validation_factor)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.caracts_count, arhiteche, window_size)
    autoencoder.build((1, window_size, training_dataset.caracts_count))
    autoencoder.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        init_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    print("Автоэнкодер определён.")

    if continue_education:
        checkpoint_name = "modeles\\EventAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_" + str(checkpoint)
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
    print("Запускаем обучение!")

    arhiteche = {"GRU_1": (30, 34), "GRU_2": (27, 30), "GRU_3": (25, 27), "GRU_4": (23, 25), "GRU_5": (20, 23),
                 "GRU_6": (23, 20), "GRU_7": (25, 23), "GRU_8": (27, 25), "GRU_9": (30, 27), "GRU_A": (34, 30)}
    versia = "0.3.4"
    print("\n\n" + versia)
    main(versia, arhiteche)