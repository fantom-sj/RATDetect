import subprocess as sp


def main():
    try:
        process = sp.run(["python", "EventAD_Train_RNN.py"], check=True)
    except sp.CalledProcessError as e:
        print(f"Ошибка во время обучения!")
        print("Перезапуск процесса обучения")
        main()


if __name__ == '__main__':
    print("Запуск процесса обучения!")
    main()