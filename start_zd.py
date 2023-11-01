# Development time：2023/4/17  23:13
import getdata as gd
import calculate as cal
import numpy as np
import foldPrice as fP
import new_cal as n_cal
import new_fold as n_fold
from scipy.spatial.distance import directed_hausdorff
import math
import matplotlib.pyplot as plt
import tensorflow as tf


def fun_zd(pri, dt, poi, f_zhi):
    id_all = []
    lg = 0
    pri_lg = 0
    for f in range(len(dt['face']) - 1, -1, -1):
        # Take the first fold and fold it with the least cost
        if f == pri[0][0] and pri[0][1] < 10000000000.0:
            id1 = int(dt['face'][f][0].split('/')[0]) - 1
            id2 = int(dt['face'][f][1].split('/')[0]) - 1
            id3 = int(dt['face'][f][2].split('/')[0]) - 1
            id_all.append(id1)
            id_all.append(id2)
            id_all.append(id3)
            for po_id in range(len(poi)):
                if f == poi[po_id][1]:
                    new_point = [poi[po_id][0][0], poi[po_id][0][1], poi[po_id][0][2]]
                    id_all.sort()
                    dt["vertex"][id_all[0]] = new_point
                    dt["类型"][id_all[0]] = poi[po_id][2]
                    del dt["类型"][id_all[1]]
                    del dt["类型"][id_all[2] - 1]
                    for fid in range(len(dt['face']) - 1, -1, -1):
                        id11 = int(dt['face'][fid][0].split('/')[0]) - 1
                        id12 = int(dt['face'][fid][1].split('/')[0]) - 1
                        id13 = int(dt['face'][fid][2].split('/')[0]) - 1
                        set1 = {id11, id12}
                        set2 = {id11, id13}
                        set3 = {id12, id13}
                        if set1.issubset(id_all) or set2.issubset(id_all) or set3.issubset(id_all):
                            dt['face'].remove(dt['face'][fid])
                            del f_zhi[0][fid]
                            del f_zhi[1][fid]
                            del f_zhi[2][fid]
                            for pr in range(len(pri) - 1, -1, -1):
                                if pri[pr][0] == fid:
                                    del pri[pr]
                                elif pri[pr][0] > fid:
                                    pri[pr][0] = pri[pr][0] - 1
                            for pr1 in range(len(poi) - 1, -1, -1):
                                if poi[pr1][1] == fid:
                                    del poi[pr1]
                                elif poi[pr1][1] > fid:
                                    poi[pr1][1] = poi[pr1][1] - 1
                        elif id11 in id_all:
                            dt["face"][fid].append(0)
                            lg = lg + 1
                            pri_lg = pri_lg + 1
                            if id11 != id_all[0]:
                                dt['face'][fid][0] = dt['face'][fid][0].replace(
                                    dt['face'][fid][0].split('/')[0],
                                    str(id_all[0] + 1), 1)
                            if id12 > id_all[1]:
                                if id12 < id_all[2]:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12), 1)
                                else:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12 - 1), 1)
                            if id13 > id_all[1]:
                                if id13 < id_all[2]:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13), 1)
                                else:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13 - 1), 1)
                        elif id12 in id_all:
                            dt["face"][fid].append(0)
                            lg = lg + 1
                            pri_lg = pri_lg + 1
                            if id12 != id_all[0]:
                                dt['face'][fid][1] = dt['face'][fid][1].replace(
                                    dt['face'][fid][1].split('/')[0],
                                    str(id_all[0] + 1), 1)
                            if id11 > id_all[1]:
                                if id11 < id_all[2]:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11), 1)
                                else:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11 - 1), 1)
                            if id13 > id_all[1]:
                                if id13 < id_all[2]:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13), 1)
                                else:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13 - 1), 1)
                        elif id13 in id_all:
                            dt["face"][fid].append(0)
                            lg = lg + 1
                            pri_lg = pri_lg + 1
                            if id13 != id_all[0]:
                                dt['face'][fid][2] = dt['face'][fid][2].replace(
                                    dt['face'][fid][2].split('/')[0],
                                    str(id_all[0] + 1), 1)
                            if id11 > id_all[1]:
                                if id11 < id_all[2]:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11), 1)
                                else:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11 - 1), 1)
                            if id12 > id_all[1]:
                                if id12 < id_all[2]:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12), 1)
                                else:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12 - 1), 1)
                        else:
                            if id11 > id_all[1]:
                                if id11 < id_all[2]:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11), 1)
                                else:
                                    dt['face'][fid][0] = dt['face'][fid][0].replace(
                                        dt['face'][fid][0].split('/')[0]
                                        , str(id11 - 1), 1)
                            if id12 > id_all[1]:
                                if id12 < id_all[2]:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12), 1)
                                else:
                                    dt['face'][fid][1] = dt['face'][fid][1].replace(
                                        dt['face'][fid][1].split('/')[0]
                                        , str(id12 - 1), 1)
                            if id13 > id_all[1]:
                                if id13 < id_all[2]:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13), 1)
                                else:
                                    dt['face'][fid][2] = dt['face'][fid][2].replace(
                                        dt['face'][fid][2].split('/')[0]
                                        , str(id13 - 1), 1)
                    for fid in range(len(dt['face'])):
                        if len(dt['face'][fid]) == 4:
                            id11 = int(dt['face'][fid][0].split('/')[0]) - 1
                            id12 = int(dt['face'][fid][1].split('/')[0]) - 1
                            id13 = int(dt['face'][fid][2].split('/')[0]) - 1
                            id_arr = [id11, id12, id13]
                            for fid1 in range(len(dt['face'])):
                                id11_ = int(dt['face'][fid][0].split('/')[0]) - 1
                                id12_ = int(dt['face'][fid][1].split('/')[0]) - 1
                                id13_ = int(dt['face'][fid][2].split('/')[0]) - 1
                                if (id11_ in id_arr or id12_ in id_arr or id13_ in id_arr) and len(dt['face'][fid1]
                                                                                                   ) != 4:
                                    dt["face"][fid1].append(1)
                                    pri_lg = pri_lg + 1
                    del dt['vertex'][id_all[1]]
                    del dt['vertex'][id_all[2] - 1]
                    break
            break
    id_arr = []
    # Remove the same face index
    for f_ in range(len(dt['face']) - 1, -1, -1):
        id11 = int(dt['face'][f_][0].split('/')[0]) - 1
        id22 = int(dt['face'][f_][1].split('/')[0]) - 1
        id33 = int(dt['face'][f_][2].split('/')[0]) - 1
        id_arr_ = [id11, id22, id33]
        id_arr_.sort()
        if len(id_arr) != 0:
            for i in id_arr:
                if i[0] == id_arr_:
                    if len(dt['face'][f_]) == 4 and len(dt['face'][i[1]]) == 4:
                        if dt['face'][i[1]][3] == 0:
                            lg = lg - 1
                        else:
                            pri_lg = pri_lg - 1
                        del dt['face'][f_]
                        del f_zhi[0][f_]
                        del f_zhi[1][f_]
                        del f_zhi[2][f_]
                        for pr, pr1 in zip(range(len(pri) - 1, -1, -1), range(len(poi) - 1, -1, -1)):
                            if pri[pr][0] == f_:
                                del pri[pr]
                            elif pri[pr][0] > f_:
                                pri[pr][0] = pri[pr][0] - 1
                            if poi[pr1][1] == f_:
                                del poi[pr1]
                            elif poi[pr1][1] > f_:
                                poi[pr1][1] = poi[pr1][1] - 1
                    elif len(dt['face'][f_]) == 4:
                        del dt['face'][i[1]]
                        del f_zhi[0][i[1]]
                        del f_zhi[1][i[1]]
                        del f_zhi[2][i[1]]
                        for pr, pr1 in zip(range(len(pri) - 1, -1, -1), range(len(poi) - 1, -1, -1)):
                            if pri[pr][0] == i[1]:
                                del pri[pr]
                            elif pri[pr][0] > i[1]:
                                pri[pr][0] = pri[pr][0] - 1
                            if poi[pr1][1] == i[1]:
                                del poi[pr1]
                            elif poi[pr1][1] > i[1]:
                                poi[pr1][1] = poi[pr1][1] - 1
                    else:
                        del dt['face'][f_]
                        del f_zhi[0][f_]
                        del f_zhi[1][f_]
                        del f_zhi[2][f_]
                        for pr, pr1 in zip(range(len(pri) - 1, -1, -1), range(len(poi) - 1, -1, -1)):
                            if pri[pr][0] == f_:
                                del pri[pr]
                            elif pri[pr][0] > f_:
                                pri[pr][0] = pri[pr][0] - 1
                            if poi[pr1][1] == f_:
                                del poi[pr1]
                            elif poi[pr1][1] > f_:
                                poi[pr1][1] = poi[pr1][1] - 1
        id_arr.append([id_arr_, f_])
        if lg <= 0:
            lg = pri_lg
    return dt, pri, poi, f_zhi, lg, pri_lg


def update(file, file1, file2, file3, file4, file5, dis1, dis2, dis3, dis4):
    with tf.device('/gpu:0'):
        with tf.compat.v1.Session() as sess:
            jhl = 0
            u = []
            data = gd.get_obj(file)
            data, face_zhi, length = cal.calculate(data, np, dis1, dis2, dis3, dis4)
            for dvt in data['vertex']:
                u.append(tuple(dvt))
            price, point = fP.fold(data, face_zhi, np, math)
            while jhl <= 0.5:
                data, price, point, face_zhi, lg_, pri_lg = fun_zd(price, data, point, face_zhi)
                if price[0][1] >= 10000000000.0:
                    # print(price[0][1])
                    break
                else:
                    face_zhi = n_cal.new_calculate(data, face_zhi, np, lg_)
                    data, price, point = n_fold.new_fold(data, face_zhi, price, point, np, math, pri_lg)
                    # print(len(data['face']))
                    jhl = 1 - len(data['face']) / length
                    if 0.1 <= jhl <= 0.103:
                        v = []
                        for dvt in data['vertex']:
                            v.append(tuple(dvt))
                        qh_zhi1 = round(max(directed_hausdorff(np.array(u), np.array(v))[0],
                                            directed_hausdorff(np.array(v), np.array(u))[0]), 4)
                        file = open(file1, 'w')
                        for ver in data['vertex']:
                            v_list = ['v ', str(ver[0]) + ' ', str(ver[1]) + ' ', str(ver[2]) + '\n']
                            file.writelines(v_list)
                        for fa in data['face']:
                            f_list = ['f ', fa[0] + ' ', fa[1] + ' ', fa[2] + '\n']
                            file.writelines(f_list)
                    if 0.2 <= jhl <= 0.203:
                        v = []
                        for dvt in data['vertex']:
                            v.append(tuple(dvt))
                        qh_zhi2 = round(max(directed_hausdorff(np.array(u), np.array(v))[0],
                                            directed_hausdorff(np.array(v), np.array(u))[0]), 4)
                        file = open(file2, 'w')
                        for ver in data['vertex']:
                            v_list = ['v ', str(ver[0]) + ' ', str(ver[1]) + ' ', str(ver[2]) + '\n']
                            file.writelines(v_list)
                        for fa in data['face']:
                            f_list = ['f ', fa[0] + ' ', fa[1] + ' ', fa[2] + '\n']
                            file.writelines(f_list)
                    if 0.3 <= jhl <= 0.303:
                        v = []
                        for dvt in data['vertex']:
                            v.append(tuple(dvt))
                        qh_zhi3 = round(max(directed_hausdorff(np.array(u), np.array(v))[0],
                                            directed_hausdorff(np.array(v), np.array(u))[0]), 4)
                        file = open(file3, 'w')
                        for ver in data['vertex']:
                            v_list = ['v ', str(ver[0]) + ' ', str(ver[1]) + ' ', str(ver[2]) + '\n']
                            file.writelines(v_list)
                        for fa in data['face']:
                            f_list = ['f ', fa[0] + ' ', fa[1] + ' ', fa[2] + '\n']
                            file.writelines(f_list)
                    if 0.4 <= jhl <= 0.403:
                        v = []
                        for dvt in data['vertex']:
                            v.append(tuple(dvt))
                        qh_zhi4 = round(max(directed_hausdorff(np.array(u), np.array(v))[0],
                                            directed_hausdorff(np.array(v), np.array(u))[0]), 4)
                        file = open(file4, 'w')
                        for ver in data['vertex']:
                            v_list = ['v ', str(ver[0]) + ' ', str(ver[1]) + ' ', str(ver[2]) + '\n']
                            file.writelines(v_list)
                        for fa in data['face']:
                            f_list = ['f ', fa[0] + ' ', fa[1] + ' ', fa[2] + '\n']
                            file.writelines(f_list)
                        # print(data)
            v = []
            for dvt in data['vertex']:
                v.append(tuple(dvt))
            qh_zhi5 = round(max(directed_hausdorff(np.array(u), np.array(v))[0],
                                directed_hausdorff(np.array(v), np.array(u))[0]), 4)
            y_axis_data = [qh_zhi1, qh_zhi2, qh_zhi3, qh_zhi4, qh_zhi5]
            file = open(file5, 'w')
            for ver in data['vertex']:
                v_list = ['v ', str(ver[0]) + ' ', str(ver[1]) + ' ', str(ver[2]) + '\n']
                file.writelines(v_list)
            for fa in data['face']:
                f_list = ['f ', fa[0] + ' ', fa[1] + ' ', fa[2] + '\n']
                file.writelines(f_list)
            return y_axis_data


y_data1 = update('model1.obj', 'model1_90.obj', 'model1_80.obj', 'model1_70.obj', 'model1_60.obj', 'model1_50.obj',
                 10, 10, 5, 10)
y_data2 = update('model2.obj', 'model2_90.obj', 'model2_80.obj', 'model2_70.obj', 'model2_60.obj', 'model2_50.obj',
                 25, 5, 8, 1)
y_data3 = update('model3.obj', 'model3_90.obj', 'model3_80.obj', 'model3_70.obj', 'model3_60.obj', 'model3_50.obj',
                 10, 3, 10, 1)
x_data = ['10%', '20%', '30%', '40%', '50%']
plt.plot(x_data, y_data1, 'b*--', alpha=0.5, linewidth=1, label='model1')
plt.plot(x_data, y_data2, 'rs--', alpha=0.5, linewidth=1, label='model2')
plt.plot(x_data, y_data3, 'go--', alpha=0.5, linewidth=1, label='model3')

for a, b in zip(x_data, y_data1):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)
for a, b1 in zip(x_data, y_data2):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
for a, b2 in zip(x_data, y_data3):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
plt.legend()
plt.xlabel('Simplification Rate')
plt.ylabel('Hausdorff Distance')
plt.show()

for x, y1 in zip(x_data, y_data1):
    plt.text(x, y1, str(y1), ha='center', va='bottom', fontsize=8)
plt.plot(x_data, y_data1, 'b*--', alpha=0.5, linewidth=1, label='model1')
plt.legend()
plt.xlabel('Simplification Rate')
plt.ylabel('Hausdorff Distance')
plt.show()

for x, y2 in zip(x_data, y_data2):
    plt.text(x, y2, str(y2), ha='center', va='bottom', fontsize=8)
plt.plot(x_data, y_data2, 'rs--', alpha=0.5, linewidth=1, label='model2')
plt.legend()
plt.xlabel('Simplification Rate')
plt.ylabel('Hausdorff Distance')
plt.show()

for x, y3 in zip(x_data, y_data3):
    plt.text(x, y3, str(y3), ha='center', va='bottom', fontsize=8)
plt.plot(x_data, y_data3, 'go--', alpha=0.5, linewidth=1, label='model3')
plt.legend()
plt.xlabel('Simplification Rate')
plt.ylabel('Hausdorff Distance')
plt.show()





