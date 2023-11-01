# Development time：2023/4/12  8:49
def get_qmd(np, data, id_all, point, q_md, math, cvi1, cvi2, cvi3, hgi1, hgi2, hgi3):
    # Calculate the Mahalanobis distance
    xcd_arr = []
    tj_arr = []
    f_arr = []
    v_arr = []
    angle_ = 0
    normals = np.array([0, 0, 0])
    for face1, fi in zip(data['face'], range(len(data['face']))):
        id11 = int(face1[0].split('/')[0]) - 1
        id12 = int(face1[1].split('/')[0]) - 1
        id13 = int(face1[2].split('/')[0]) - 1
        id_arr = [id11, id12, id13]
        count = [x for x in id_all if x in id_arr]
        if (id11 in id_all or id12 in id_all or id13 in id_all) and len(count) == 1:
            if count[0] == id11:
                d1 = np.array(point)
                d2 = np.array(data['vertex'][id12])
                d3 = np.array(data['vertex'][id13])
                b1 = np.linalg.norm(d1 - d2)
                b2 = np.linalg.norm(d1 - d3)
                b3 = np.linalg.norm(d2 - d3)
                if np.linalg.norm(np.cross((d1 - d2), (d1 - d3))) == 0:
                    point = point + np.array([0.000001, 0.000001, 0.000001])
                    d1 = np.array(point)
                cos = np.clip(np.dot((d1 - d2), (d1 - d3)) / (b1 * b2), -1, 1)
                angle_ = angle_ + np.arccos(cos)
            elif count[0] == id12:
                d1 = np.array(data['vertex'][id11])
                d2 = np.array(point)
                d3 = np.array(data['vertex'][id13])
                b1 = np.linalg.norm(d1 - d2)
                b2 = np.linalg.norm(d1 - d3)
                b3 = np.linalg.norm(d2 - d3)
                if np.linalg.norm(np.cross((d1 - d2), (d1 - d3))) == 0:
                    point = point + np.array([0.000001, 0.000001, 0.000001])
                    d2 = np.array(point)
                cos = np.clip(np.dot((d1 - d2), (d1 - d3)) / (b1 * b2), -1, 1)
                angle_ = angle_ + np.arccos(cos)
            else:
                d1 = np.array(data['vertex'][id11])
                d2 = np.array(data['vertex'][id12])
                d3 = np.array(point)
                b1 = np.linalg.norm(d1 - d2)
                b2 = np.linalg.norm(d1 - d3)
                b3 = np.linalg.norm(d2 - d3)
                if np.linalg.norm(np.cross((d1 - d2), (d1 - d3))) == 0:
                    point = point + np.array([0.000001, 0.000001, 0.000001])
                    d3 = np.array(point)
                cos = np.clip(np.dot((d1 - d2), (d1 - d3)) / (b1 * b2), -1, 1)
                angle_ = angle_ + np.arccos(cos)
            v = np.cross((d1-d2), (d1-d3))
            if v[1] < 0:
                v = -v
            length = np.sqrt(v.dot(v))
            v = v / length
            v_arr.append(v)
            s = (b1 + b2 + b3) / 2
            area = np.sqrt(s * (s - b1) * (s - b2) * (s - b3))
            if area == 0:
                b_arr = [b1, b2, b3]
                b_arr.sort()
                if b1 == b_arr[0]:
                    b1 = b1 + 0.001
                elif b2 == b_arr[0]:
                    b2 = b2 + 0.001
                elif b3 == b_arr[0]:
                    b3 = b3 + 0.001
                s = (b1 + b2 + b3) / 2
                area = np.sqrt(s * (s - b1) * (s - b2) * (s - b3))
            f_arr.append(area)
            normals = normals+np.dot(area, v)
            face_jz = np.array([d1, d2, d3])
            tj_arr.append(np.linalg.det(face_jz))
            v1 = d1 - d2
            v2 = d1 - d3
            v11 = d2 - d1
            v22 = d2 - d3
            v111 = d3 - d1
            v222 = d3 - d2
            arc1 = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            angle1 = np.arccos(arc1)
            arc2 = np.clip(np.dot(v11, v22) / (np.linalg.norm(v11) * np.linalg.norm(v22)), -1, 1)
            angle2 = np.arccos(arc2)
            arc3 = np.clip(np.dot(v111, v222) / (np.linalg.norm(v111) * np.linalg.norm(v222)), -1, 1)
            angle3 = np.arccos(arc3)
            angle = min(angle1, angle2, angle3)
            xcd_arr.append(1 - np.cos(angle))
    tj = sum(tj_arr)
    xcd = sum(xcd_arr) / len(xcd_arr)
    cvi = 0
    for vi in v_arr:
        cvi = cvi + np.arccos(np.clip(np.dot(normals / np.linalg.norm(normals), vi), -1, 1))
    cvi = cvi / len(f_arr)
    hgi = (2 * math.pi - angle_) / sum(f_arr)
    x = np.array([[xcd, 0.5], [q_md[0], tj], [q_md[1], 0], [q_md[2], cvi], [q_md[3], hgi], [abs(cvi1), 0],
                  [abs(cvi2), 0], [abs(cvi3), 0], [abs(hgi1), 0], [abs(hgi2), 0], [abs(hgi3), 0]])
    mean = np.mean(x, axis=0)
    x_mean = x - mean
    std = np.std(x_mean, axis=0)
    x_ = x_mean / std
    d_ = np.cov(x_.T)
    inv = np.linalg.inv(d_)
    qmz = np.dot(np.dot(x_, inv), x_.T)
    return sum(qmz.diagonal()) / len(qmz.diagonal()), point


def get_ql(np, data, d_id1, d_id2, d_id3, angle, result, fid):
    # Calculate vertex importance
    d1 = np.array(data['vertex'][d_id1])
    d2 = np.array(data['vertex'][d_id2])
    d3 = np.array(data['vertex'][d_id3])
    lx1 = data['类型'][d_id1]
    lx2 = data['类型'][d_id2]
    lx3 = data['类型'][d_id3]
    bj_id = -1
    if lx1 == '普通顶点' and lx2 == '普通顶点' and lx3 == '普通顶点':
        lx = '普通三角形'
    elif lx1 == '普通顶点' and lx2 == '普通顶点' and lx3 == '边界顶点':
        lx = '边界三角形'
        bj_id = d_id3
    elif lx1 == '普通顶点' and lx3 == '普通顶点' and lx2 == '边界顶点':
        lx = '边界三角形'
        bj_id = d_id2
    elif lx2 == '普通顶点' and lx3 == '普通顶点' and lx1 == '边界顶点':
        lx = '边界三角形'
        bj_id = d_id1
    else:
        lx = '其它三角形'
    b1 = np.linalg.norm(d1 - d2)
    b2 = np.linalg.norm(d1 - d3)
    cos = np.clip(np.dot((d1-d2), (d1-d3))/(b1*b2), -1, 1)
    angle = angle + np.arccos(cos)
    if len(result) != 0:
        for re in result:
            if re[0] != fid:
                result.append([fid, lx, bj_id])
                break
    else:
        result.append([fid, lx, bj_id])
    return angle, result


def fold(data, face_zhi, np, math):
    zd_price_arr = []
    new_point_arr = []
    for face, fid in zip(data['face'], range(len(data['face']))):
        id1 = int(face[0].split('/')[0]) - 1
        id2 = int(face[1].split('/')[0]) - 1
        id3 = int(face[2].split('/')[0]) - 1
        id_all = [id1, id2, id3]
        face_delete = []
        face_delete12 = []
        face_delete13 = []
        face_delete23 = []
        face_xg = []
        face_xg1 = []
        face_xg2 = []
        face_xg3 = []
        q_ed1 = np.zeros((4, 4))
        q_ed2 = np.zeros((4, 4))
        q_ed3 = np.zeros((4, 4))
        mj = face_zhi[1][fid]
        tj = face_zhi[2][fid]
        kvi_1 = np.array([0, 0, 0])
        kvi_2 = np.array([0, 0, 0])
        kvi_3 = np.array([0, 0, 0])
        s1 = 0
        s2 = 0
        s3 = 0
        angle1 = 0
        angle2 = 0
        angle3 = 0
        arr1 = []
        arr2 = []
        arr3 = []
        res_arr = []
        for f in range(len(data["face"])):
            dian_id1 = int(data["face"][f][0].split('/')[0]) - 1
            dian_id2 = int(data["face"][f][1].split('/')[0]) - 1
            dian_id3 = int(data["face"][f][2].split('/')[0]) - 1
            dian_id_arr = [dian_id1, dian_id2, dian_id3]
            dian_jj = [x for x in id_all if x in dian_id_arr]

            if id1 in dian_id_arr:
                q_ed1 = q_ed1 + face_zhi[0][f][1]
                s1 = s1+face_zhi[1][f]
                kvi_1 = kvi_1+np.dot(face_zhi[1][f], face_zhi[0][f][0])
                arr1.append(face_zhi[0][f][0])
                if dian_id1 == id1:
                    angle1, res_arr = get_ql(np, data, dian_id1, dian_id2, dian_id3, angle1, res_arr, f)
                elif dian_id2 == id1:
                    angle1, res_arr = get_ql(np, data, dian_id2, dian_id1, dian_id3, angle1, res_arr, f)
                elif dian_id3 == id1:
                    angle1, res_arr = get_ql(np, data, dian_id3, dian_id1, dian_id2, angle1, res_arr, f)
            if id2 in dian_id_arr:
                q_ed2 = q_ed2 + face_zhi[0][f][1]
                s2 = s2 + face_zhi[1][f]
                kvi_2 = kvi_2 + np.dot(face_zhi[1][f], face_zhi[0][f][0])
                arr2.append(face_zhi[0][f][0])
                if dian_id1 == id2:
                    angle2, res_arr = get_ql(np, data, dian_id1, dian_id2, dian_id3, angle2, res_arr, f)
                elif dian_id2 == id2:
                    angle2, res_arr = get_ql(np, data, dian_id2, dian_id1, dian_id3, angle2, res_arr, f)
                elif dian_id3 == id2:
                    angle2, res_arr = get_ql(np, data, dian_id3, dian_id1, dian_id2, angle2, res_arr, f)
            if id3 in dian_id_arr:
                q_ed3 = q_ed3 + face_zhi[0][f][1]
                s3 = s3 + face_zhi[1][f]
                kvi_3 = kvi_3 + np.dot(face_zhi[1][f], face_zhi[0][f][0])
                arr3.append(face_zhi[0][f][0])
                if dian_id1 == id3:
                    angle3, res_arr = get_ql(np, data, dian_id1, dian_id2, dian_id3, angle3, res_arr, f)
                elif dian_id2 == id3:
                    angle3, res_arr = get_ql(np, data, dian_id2, dian_id1, dian_id3, angle3, res_arr, f)
                elif dian_id3 == id3:
                    angle3, res_arr = get_ql(np, data, dian_id3, dian_id1, dian_id2, angle3, res_arr, f)
            # Computes information about a triangle that has two vertices adjacent to the folded triangle
            if len(dian_jj) == 2:
                face_delete.append(f)
                tj = tj + face_zhi[2][f]
                mj = mj + face_zhi[1][f]
                if (dian_jj[0] == id1 and dian_jj[1] == id2) or (dian_jj[0] == id2 and dian_jj[1] == id1):
                    face_delete12.append(f)
                elif (dian_jj[0] == id1 and dian_jj[1] == id3) or (dian_jj[0] == id3 and dian_jj[1] == id1):
                    face_delete13.append(f)
                else:
                    face_delete23.append(f)
            # Computes information about triangles that have only one vertex adjacent to the folded triangle
            if len(dian_jj) == 1:
                face_xg.append(f)
                tj = tj + face_zhi[2][f]
                mj = mj + face_zhi[1][f]
                if dian_jj[0] == id1:
                    face_xg1.append(f)
                elif dian_jj[0] == id2:
                    face_xg2.append(f)
                else:
                    face_xg3.append(f)
        jzz = np.array([0, 0, 0, 1])
        q_ed = q_ed1 + q_ed2 + q_ed3
        q_ed1[[3], :] = jzz
        q_ed2[[3], :] = jzz
        q_ed3[[3], :] = jzz
        cvi_1 = 0
        cvi_2 = 0
        cvi_3 = 0
        for a1 in arr1:
            cvi_1 = cvi_1 + np.arccos(np.clip(np.dot(kvi_1/np.linalg.norm(kvi_1), a1), -1, 1))
        for a2 in arr2:
            cvi_2 = cvi_2 + np.arccos(np.clip(np.dot(kvi_2/np.linalg.norm(kvi_2), a2), -1, 1))
        for a3 in arr3:
            cvi_3 = cvi_3 + np.arccos(np.clip(np.dot(kvi_3/np.linalg.norm(kvi_3), a3), -1, 1))
        cvi_1 = cvi_1 / len(arr1)
        cvi_2 = cvi_2 / len(arr2)
        cvi_3 = cvi_3 / len(arr3)
        cvi = (cvi_1 + cvi_2 + cvi_3)/3
        hgi1 = (2 * math.pi - angle1) / s1
        hgi2 = (2 * math.pi - angle2) / s2
        hgi3 = (2 * math.pi - angle3) / s3
        hgi = (hgi1 + hgi2 + hgi3) / 3
        mj = mj / (len(face_delete) + len(face_xg))
        q_md = [tj, mj, cvi, hgi]
        for r in res_arr:
            if fid == r[0]:
                bj_id = r[2]
                lx = r[1]
        # Folded ordinary triangle
        if lx == '普通三角形':
            v1_qz1 = 0
            v2_qz2 = 0
            v3_qz3 = 0
            v12_qz = 0
            v13_qz = 0
            v23_qz = 0
            # Calculate the weights of the points separately
            for fx1_ in face_xg1:
                v1_qz1 = v1_qz1 + (1 - abs(np.dot(face_zhi[0][fx1_][0], face_zhi[0][fid][0]))) / face_zhi[1][fx1_]
            for fx2_ in face_xg2:
                v2_qz2 = v2_qz2 + (1 - abs(np.dot(face_zhi[0][fx2_][0], face_zhi[0][fid][0]))) / face_zhi[1][fx2_]
            for fx3_ in face_xg3:
                v3_qz3 = v3_qz3 + (1 - abs(np.dot(face_zhi[0][fx3_][0], face_zhi[0][fid][0]))) / face_zhi[1][fx3_]
            for fd1_ in face_delete12:
                v12_qz = v12_qz + (1 - abs(np.dot(face_zhi[0][fd1_][0], face_zhi[0][fid][0]))) / face_zhi[1][fd1_]
            for fd2_ in face_delete13:
                v13_qz = v13_qz + (1 - abs(np.dot(face_zhi[0][fd2_][0], face_zhi[0][fid][0]))) / face_zhi[1][fd2_]
            for fd3_ in face_delete23:
                v23_qz = v23_qz + (1 - abs(np.dot(face_zhi[0][fd3_][0], face_zhi[0][fid][0]))) / face_zhi[1][fd3_]
            v1_qz = (v1_qz1 + 0.5 * v12_qz + 0.5 * v13_qz) / (v1_qz1 + v2_qz2 + v3_qz3 + v12_qz + v13_qz + v23_qz)
            v2_qz = (v2_qz2 + 0.5 * v12_qz + 0.5 * v23_qz) / (v1_qz1 + v2_qz2 + v3_qz3 + v12_qz + v13_qz + v23_qz)
            v3_qz = 1 - v1_qz - v2_qz

            if np.linalg.det(q_ed1) != 0.0:
                arr_point1 = np.matmul(np.linalg.inv(q_ed1), jzz.T)
            else:
                arr_point1 = data['vertex'][id1]
            if np.linalg.det(q_ed2) != 0.0:
                arr_point2 = np.matmul(np.linalg.inv(q_ed2), jzz.T)
            else:
                arr_point2 = data['vertex'][id2]
            if np.linalg.det(q_ed3) != 0.0:
                arr_point3 = np.matmul(np.linalg.inv(q_ed3), jzz.T)
            else:
                arr_point3 = data['vertex'][id3]
            # The vertex coordinates after folding are optimized
            x0 = v1_qz * (arr_point1[0]) + v2_qz * (arr_point2[0]) + v3_qz * (arr_point3[0])
            y0 = v1_qz * (arr_point1[1]) + v2_qz * (arr_point2[1]) + v3_qz * (arr_point3[1])
            z0 = v1_qz * (arr_point1[2]) + v2_qz * (arr_point2[2]) + v3_qz * (arr_point3[2])
            point = [x0, y0, z0]
            qm_zhi, point = get_qmd(np, data, id_all, point, q_md, math, cvi_1, cvi_2, cvi_3, hgi1, hgi2, hgi3, )
            arr_point = np.array([point[0], point[1], point[2], 1.0])
            zd_price = abs(np.matmul(np.matmul(arr_point, q_ed), arr_point.T)) * qm_zhi
            zd_price_arr.append(zd_price)
            new_point_arr.append([arr_point, fid, '普通顶点'])

        elif lx == '边界三角形':

            point = [data['vertex'][bj_id][0], data['vertex'][bj_id][1], data['vertex'][bj_id][2]]
            qm_zhi, point = get_qmd(np, data, id_all, point, q_md, math, cvi_1, cvi_2, cvi_3, hgi1, hgi2, hgi3, )
            arr_point = np.array([point[0], point[1], point[2], 1.0])
            zd_price = abs(np.matmul(np.matmul(arr_point, q_ed), arr_point.T)) * qm_zhi
            zd_price_arr.append(zd_price)
            new_point_arr.append([arr_point, fid, '边界顶点'])

        else:
            zd_price_arr.append(10000000000.0)
            new_point_arr.append([0, fid])
    ls = {}
    # Sort the folding costs from smallest to largest
    for zp in range(len(zd_price_arr)):
        ls[zp] = zd_price_arr[zp]
    new_price_arr_all = sorted(ls.items(), key=lambda x: x[1])
    for pa in range(len(new_price_arr_all)):
        new_price_arr_all[pa] = list(new_price_arr_all[pa])
    return new_price_arr_all, new_point_arr










