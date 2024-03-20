from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
app = Flask(__name__)
import matplotlib
matplotlib.use('agg')

col = 20

#Считаем нужные матрицы:
dfA0 = pd.read_csv('data/A0.csv')
A_0 = np.array(dfA0)
A_0 = A_0.transpose()[1:].transpose()

dfA1 = pd.read_csv('data/A1.csv')
A_1 = np.array(dfA1)
A_1 = A_1.transpose()[1:].transpose()

dfG0 = pd.read_csv('data/G0.csv')
G_0 = np.array(dfG0)
G_0 = G_0.transpose()[1:].transpose()

dfG1 = pd.read_csv('data/G1.csv')
G_1 = np.array(dfG1)
G_1 = G_1.transpose()[1:].transpose()

dfVIP0 = pd.read_csv('data/VIP0.csv')
VIP_0 = np.array(dfVIP0)
VIP_0 = VIP_0.transpose()[1:].transpose()
VIP_0 = VIP_0.reshape(-1)

dfVIP1 = pd.read_csv('data/VIP1.csv')
VIP_1 = np.array(dfVIP1)
VIP_1 = VIP_1.transpose()[1:].transpose()
VIP_1 = VIP_1.reshape(-1)

dfY0 = pd.read_csv('data/Y0.csv')
Y_0 = np.array(dfY0)
Y_0 = Y_0.transpose()[1:].transpose()
Y_0 = Y_0.reshape(-1)

dfY1 = pd.read_csv('data/Y1.csv')
Y_1 = np.array(dfY1)
Y_1 = Y_1.transpose()[1:].transpose()
Y_1 = Y_1.reshape(-1)

dfInd = pd.read_csv('data/ind.csv')
ind_agr = dfInd['0'].to_list()

dfCou = pd.read_csv('data/cou.csv')
cou_agr = dfCou['0'].to_list()


@app.route('/')
def index():
    return render_template('calculator.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    num1 = float(request.form['num1'])
    num2 = float(request.form['num2'])
    operator = request.form['operator']

    result = 0
    if operator == 'add':
        result = num1 + num2
    elif operator == 'subtract':
        result = num1 - num2
    elif operator == 'multiply':
        result = num1 * num2
    elif operator == 'divide':
        if num2 != 0:
            result = num1 / num2
        else:
            return "Error: Cannot divide by zero!"

    return render_template('calculator.html', result=result)


@app.route('/main')
def calc():
    return render_template('gpt_temp.html')


@app.route('/run_code', methods=['GET', 'POST'])
def run_code():
    # Получение данных от клиента
    scen_otr = int(request.form['sector'])
    scen_v = float(request.form['import_change'])
    scen_cou_to = int(request.form['country'])
    scen_year = int(request.form['year'])
    scen_n = int(request.form['scenario'])

    # Ваш Python-код здесь...
    from numpy.linalg import inv
    import matplotlib.pyplot as plt

    def custom_inv(Mat):  # для нормального нахождения обратной матрицы
        Mat_x = 1.5 * Mat
        inv_Mat = np.linalg.inv(Mat_x)
        return inv_Mat / (2 / 3)

    # сортировка - вспом функция
    def sort(X, Y):
        # отсортируем по возрастанию
        for i in range(len(Y) - 1):
            for j in range(len(Y) - i - 1):
                if Y[j] > Y[j + 1]:
                    Y[j], Y[j + 1] = Y[j + 1], Y[j]
                    X[j], X[j + 1] = X[j + 1], X[j]

    def findOut(A, G, Y):  # функция нахождения выпусков
        L = inv(custom_inv(G) - A)
        X = np.dot(L, Y.transpose())#
        return X

    def sort_mod(X, Y):
        # отсортируем по возрастанию модуля
        for i in range(len(Y) - 1):
            for j in range(len(Y) - i - 1):
                if abs(Y[j]) > abs(Y[j + 1]):
                    Y[j], Y[j + 1] = Y[j + 1], Y[j]
                    X[j], X[j + 1] = X[j + 1], X[j]

    # 2 - уменьшение/увеличение имп из ост мира
    if (scen_year ==1):
        G_new = np.array(G_0)
        A_scen = np.array(A_0)
        Y_scen = np.array(Y_0)
        VIP_scen = np.array(VIP_0)
    else:
        G_new = np.array(G_1)
        A_scen = np.array(A_1)
        Y_scen = np.array(Y_1)
        VIP_scen = np.array(VIP_1)

    raz = G_new[44 * 35 + scen_otr][scen_cou_to * 35 + scen_otr] * (scen_v / 100)
    G_new[44 * 35 + scen_otr][scen_cou_to * 35 + scen_otr] = G_new[44 * 35 + scen_otr][
                                                                 scen_cou_to * 35 + scen_otr] + raz

    # 1 - сценарий
    if (scen_n == 1):
        G_new[scen_cou_to * 35 + scen_otr][scen_cou_to * 35 + scen_otr] = G_new[scen_cou_to * 35 + scen_otr][
                                                                              scen_cou_to * 35 + scen_otr] - raz

        res = findOut(A_scen, G_new, Y_scen)

    # 2 - сценарий
    if (scen_n == 2):
        sum_ost = 0
        for i in range((77 - col + 1) * 35):
            if (i == 44 * 35 + scen_otr):
                sum_ost = sum_ost
            else:
                sum_ost = sum_ost + G_new[i][scen_cou_to * 35 + scen_otr]

        for i in range((77 - col + 1)):
            if (i != 44):
                G_new[i * 35 + scen_otr][scen_cou_to * 35 + scen_otr] = raz * G_new[i * 35 + scen_otr][
                    scen_cou_to * 35 + scen_otr] / sum_ost

        res = findOut(A_scen, G_new, Y_scen)

    #print(np.transpose(G_new)[scen_cou_to * 35 + 2].sum())
    #print(raz)
    # print(G_1[scen_year - 1][44*35 + 1][scen_cou_to*35 + 1])

    # вывод графиков
    # Абс изменение
    delta_VIP = (res - VIP_scen)
    y = []
    x = []
    for i in range(35):  # [44*35 + 2]
        y.append(delta_VIP[44 * 35 + i])
        x.append(ind_agr[i])

    sort(x, y)

    plt.figure(figsize=(12, 12))
    plt.title("Абсолютное изменение Выпусков Rus")  # заголовок
    plt.grid()  # включение отображение сетки
    print(y)
    plt.barh(x, y)
    plt.savefig('static/pics/gr1.png')

    y = []
    x = []
    for i in range(35):  # [44*35 + 2]
        y.append(delta_VIP[scen_cou_to * 35 + i])
        x.append(ind_agr[i])

    sort(x, y)

    plt.figure(figsize=(12, 12))
    plt.title("Абсолютное изменение Выпусков " + cou_agr[scen_cou_to])  # заголовок
    plt.grid()  # включение отображение сетки
    plt.barh(x, y)
    plt.savefig('static/pics/gr2.png')
    # Отн изменение
    delta_VIP_otn = (res - VIP_scen) / VIP_scen
    y = []
    x = []
    for i in range(35):  # [44*35 + 2]
        y.append(delta_VIP_otn[44 * 35 + i])
        x.append(ind_agr[i])

    sort(x, y)

    plt.figure(figsize=(12, 12))
    plt.title("Относительное изменение Выпусков Rus")  # заголовок
    plt.grid()  # включение отображение сетки
    plt.barh(x, y)
    plt.savefig('static/pics/gr3.png')

    y = []
    x = []
    for i in range(35):  # [44*35 + 2]
        y.append(delta_VIP_otn[scen_cou_to * 35 + i])
        x.append(ind_agr[i])

    sort(x, y)

    plt.figure(figsize=(12, 12))
    plt.title("Относительное изменение Выпусков " + cou_agr[scen_cou_to])  # заголовок
    plt.grid()  # включение отображение сетки
    plt.barh(x, y)
    plt.savefig('static/pics/gr4.png')

    # топ 10 стран по изменению выпусков
    delta_VIP_agrcou = []

    for i in range(77 - col + 1):
        delta_VIP_agrcou.append(delta_VIP[i * 35:i * 35 + 34].sum())

    y = []
    x = []
    for i in range(77 - col + 1):  # [44*35 + 2]
        y.append(delta_VIP_agrcou[i])
        x.append(cou_agr[i])

    sort_mod(x, y)

    plt.figure(figsize=(12, 12))
    plt.title("Топ 10 стран по изменению Выпусков")  # заголовок
    plt.grid()  # включение отображение сетки
    plt.bar(x[-10:], y[-10:])
    plt.savefig('static/pics/gr5.png')

    flag = 1
    # Пример возврата результата как JSON
    return render_template('gpt_temp.html', plot_url1 = "static/pics/gr1.png", plot_url2 = "static/pics/gr2.png",plot_url3 = "static/pics/gr3.png",plot_url4 = "static/pics/gr4.png",plot_url5 = "static/pics/gr5.png")


if __name__ == '__main__':
    app.run(debug=True, threaded=True)