from AutoEncoder_RNN import *
from pathlib import Path
from sklearn.ensemble import IsolationForest
import pickle

def main(versia, arhiteche, window_size):
    # Параметры датасета
    batch_size          = 1
    validation_factor   = 0.01
    feature_range       = (-1, 1)

    # Параметры оптимизатора
    # init_learning_rate  = 0.1
    # decay_steps         = 1500
    # decay_rate          = 0.96
    # staircase           = True

    # Параметры нейронной сети
    epochs              = 8
    continue_education  = True
    checkpoint          = 5
    shaffle             = True
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\TrafficAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_TAD_v" + versia
    max_min_file        = path_model + "M&M_traffic_VNAT.csv"
    dataset             = "D:\\train_characts_traffic_2.csv"
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
    # data = data[:10000]
    print(f"Загружено {len(data)} характеристик")
    data = data.sort_values(by="Flow_Charact.Time_Stamp_Start")

    # Выявленные ненужные признаки:
    # data = data.drop(["Flow_Charact.Len_Headers_Fwd"], axis=1)
    # data = data.drop(["Flow_Charact.Std_Len_Fwd_Packets"], axis=1)
    data = data.drop(["Flow_Charact.Count_Flags_URG"], axis=1)
    data = data.drop(["Flow_Charact.Count_Flags_URG_Bwd"], axis=1)
    data = data.drop(["Flow_Charact.Count_Flags_URG_Fwd"], axis=1)
    # data = data.drop(["Flow_Charact.Std_Active_Time_Flow"], axis=1)
    # data = data.drop(["Flow_Charact.Std_InActive_Time_Flow"], axis=1)
    # data = data.drop(["Flow_Charact.Std_Time_Diff_Fwd_Pkts"], axis=1)

    # print(data.to_dict("records"))
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetNetFlowTrafficGen(data, max_min_file, feature_range,
                                          batch_size, window_size, validation_factor)
    print(training_dataset.numbs_count, training_dataset.characts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.characts_count, arhiteche, window_size, batch_size)
    autoencoder.build((batch_size, window_size, training_dataset.characts_count))
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
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    print("Автоэнкодер определён.")

    if continue_education:
        checkpoint_name = "modeles\\TrafficAnomalyDetector\\" + versia + "\\Checkpoint\\" + str(checkpoint)
        autoencoder.load_weights(checkpoint_name)
        print("Продолжаем обучение:")
    else:
        checkpoint = None
        print("Начинаем обучение:")

    autoencoder.education(training_dataset, epochs=epochs, shaffle=shaffle,
                          model_checkname=path_model + "Checkpoint\\", versia=versia,
                          path_model=path_model, checkpoint=checkpoint_epoch)
    autoencoder.save(model_name)

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)


if __name__ == '__main__':
    versia = "1.6.4"

    window_size = 10

    encoder = {"1_Input": (window_size, 56), "2_GRU_seq": (45, 56), "3_GRU_seq": (35, 45),
               "4_GRU_seq": (30, 35), "5_GRU_seq": (25, 30), "6_GRU_seq": (20, 25)}
    decoder = {"7_Input": (window_size, 20), "8_GRU_seq": (25, 20), "9_GRU_seq": (30, 25),
               "10_GRU_seq": (35, 30), "11_GRU_seq": (45, 35), "12_GRU_seq": (56, 45)}
    # "6_RepeatVector": (window_size, None),

    arhiteche = (encoder, decoder)
    print("\n\n" + versia)

    with tf.name_scope("NetTraffic") as scope:
        main(versia, arhiteche, window_size)

    # path_net_traffic = f"modeles\\TrafficAnomalyDetector\\{versia}\\Checkpoint\\epoch_5"
    #
    # autoencoder = Autoencoder(56, arhiteche, window_size, 1)
    # autoencoder.build((1, window_size, 56))
    # autoencoder.summary()
    # autoencoder.encoder_model.summary()
    # autoencoder.decoder_model.summary()
    # optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # autoencoder.compile(optimizer=optimizer, loss=keras.losses.mse)
    #
    # autoencoder.load_weights(path_net_traffic)
    # autoencoder.__call__(tf.convert_to_tensor(np.random.random(56*window_size).reshape((1, window_size, 56)),
    #                                           dtype=tf.float32))
    # autoencoder.save(f"modeles\\TrafficAnomalyDetector\\{versia}\\model_TAD_v{versia}_one")