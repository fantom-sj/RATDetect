import keras.losses

from AutoEncoder_RNN import *
from pathlib import Path
from sklearn.ensemble import IsolationForest
import pickle


def main(versia, window_size, arhiteche):
    # Параметры датасета
    batch_size          = 1
    validation_factor   = 0.05
    feature_range       = (-1, 1)

    # # Параметры оптимизатора
    # init_learning_rate  = 0.001
    # decay_steps         = 50000
    # decay_rate          = 0.1
    # staircase           = True

    # Параметры нейронной сети
    epochs              = 5
    continue_education  = False
    checkpoint          = None
    checkpoint_epoch    = 0
    shaffle             = True
    loss_func           = keras.losses.mse
    arhiteche           = arhiteche
    versia              = versia
    path_model          = "modeles\\EventAnomalyDetector\\" + versia + "\\"
    model_name          = path_model + "model_EAD_v" + versia
    max_min_file        = path_model + "M&M_event.csv"
    dataset             = "D:\\train_characts_events.csv"
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
    data = data.drop(["Events_Charact.Time_Stamp_Start"], axis=1)
    data = data.drop(["Events_Charact.Time_Stamp_End"], axis=1)
    data = data.drop(["Events_Charact.Process_Name"], axis=1)
    data = data.drop(["Events_Charact.Direction_IP_Port"], axis=1)
    data = data.drop(["Events_Charact.Count_Events_Batch"], axis=1)
    data = data.drop(["Events_Charact.Duration"], axis=1)

    # Выявленные ненужные признаки:
    data = data.drop(["Events_Charact.Count_Process_Defined"], axis=1)
    data = data.drop(["Events_Charact.Count_Thread_Profile"], axis=1)
    data = data.drop(["Events_Charact.Ratio_Receive_on_Accept"], axis=1)
    data = data.drop(["Events_Charact.Ratio_Send_on_Accept"], axis=1)
    data = data.drop(["OperationName.Accept"], axis=1)
    data = data.drop(["OperationName.CreateMailSlot"], axis=1)
    data = data.drop(["OperationName.CreatePipe"], axis=1)
    data = data.drop(["OperationName.DeviceChange"], axis=1)
    data = data.drop(["OperationName.DirectoryControl"], axis=1)
    data = data.drop(["OperationName.InternalDeviceIoControl"], axis=1)
    data = data.drop(["OperationName.LockUnlockFile"], axis=1)
    data = data.drop(["OperationName.PlugAndPlay"], axis=1)
    data = data.drop(["OperationName.QueryFileQuota"], axis=1)
    data = data.drop(["OperationName.QueryInformationFile"], axis=1)
    data = data.drop(["OperationName.RenameKey"], axis=1)
    data = data.drop(["OperationName.SetFileQuota"], axis=1)
    data = data.drop(["OperationName.SetInformationFile"], axis=1)
    data = data.drop(["OperationName.SetVolumeInformation"], axis=1)
    data = data.drop(["OperationName.VolumeDismount"], axis=1)
    data = data.drop(["OperationName.VolumeMount"], axis=1)
    data = data.drop(["Events_Charact.Appeal_reg_HKCC"], axis=1)
    data = data.drop(["Events_Charact.Speed_Read_Data"], axis=1)
    data = data.drop(["Events_Charact.Speed_Write_Data"], axis=1)
    data = data.drop(["OperationName.FlushKey"], axis=1)
    data = data.drop(["OperationName.LoadKey"], axis=1)
    data = data.drop(["OperationName.QueryVolumeInformation"], axis=1)
    data = data.drop(["OperationName.SetEAFile"], axis=1)
    data = data.drop(["OperationName.SetKeySecurity"], axis=1)
    data = data.drop(["OperationName.UnloadKey"], axis=1)
    data = data.drop(["OperationName.SystemControl"], axis=1)
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, feature_range,
                                          batch_size, window_size, validation_factor)
    print(training_dataset.numbs_count, training_dataset.characts_count)

    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(training_dataset.characts_count, arhiteche, window_size)
    autoencoder.build((1, window_size, training_dataset.characts_count))
    autoencoder.summary()
    autoencoder.encoder_model.summary()
    autoencoder.decoder_model.summary()

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     init_learning_rate,
    #     decay_steps=decay_steps,
    #     decay_rate=decay_rate,
    #     staircase=staircase
    # )

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    autoencoder.compile(optimizer=optimizer, loss=loss_func)
    print("Автоэнкодер определён.")

    if continue_education:
        checkpoint_name = "modeles\\EventAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_" + checkpoint
        autoencoder.load_weights(checkpoint_name)
        print(f"Продолжаем обучение с контрольной точки: {checkpoint}")
    else:
        checkpoint = None
        print("Начинаем обучение:")

    autoencoder.education(training_dataset, epochs=epochs, shaffle=shaffle,
                          model_checkname=path_model + "Checkpoint\\", versia=versia,
                          path_model=path_model, checkpoint=checkpoint_epoch)
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
    versia = "0.7.2"
    window_size = 1

    encoder = {"1_Input": (window_size, 89), "2_LSTM_seq": (70, 89), "3_LSTM_seq": (50, 70), "4_LSTM": (25, 50)}
    decoder = {"5_RepeatVector": (window_size, None), "6_LSTM_seq": (50, 25),
               "7_LSTM_seq": (70, 50), "8_LSTM": (89, 70)}

    arhiteche = (encoder, decoder)
    print("\n\n" + versia)
    with tf.name_scope("Events") as scope:
        main(versia, window_size, arhiteche)