# Development time：2023/4/6  8:50
def calculate(data, np, dis1, dis2, dis3, dis4):
    normals = []
    area_arr = []
    tj_arr = []
    f_arr = []
    # Data cleaning, cleaning out the same face index and non-triangular face
    for f_ in range(len(data['face']) - 1, -1, -1):
        id11 = int(data['face'][f_][0].split('/')[0]) - 1
        id22 = int(data['face'][f_][1].split('/')[0]) - 1
        id33 = int(data['face'][f_][2].split('/')[0]) - 1
        if len(f_arr) != 0:
            for fi in f_arr:
                if fi == [id11, id22, id33]:
                    del data['face'][f_]
        f_arr.append([id11, id22, id33])
        if id11 == id22 or id11 == id33 or id33 == id22:
            del data['face'][f_]
    length_ = len(data['face'])
    # print(length_)
    # Calculate the unit normal vector, area, volume, Q matrix for each triangular surface
    for face in data['face']:
        id1 = int(face[0].split('/')[0]) - 1
        id2 = int(face[1].split('/')[0]) - 1
        id3 = int(face[2].split('/')[0]) - 1
        d1 = np.array(data['vertex'][id1])
        d2 = np.array(data['vertex'][id2])
        d3 = np.array(data['vertex'][id3])
        v1 = d1 - d2
        v2 = d1 - d3
        v3 = np.cross(v1, v2)
        if v3[1] < 0:
            v3 = -v3
        length = np.sqrt(v3.dot(v3))
        v = v3 / length
        x = 1 / 3 * (d1[0] + d2[0] + d3[0])
        y = 1 / 3 * (d1[1] + d2[1] + d3[1])
        z = 1 / 3 * (d1[2] + d2[2] + d3[2])
        d = -(x * v[0] + y * v[1] + z * v[2])
        p = np.array([[v[0]], [v[1]], [v[2]], [d]])
        k = np.matmul(p, p.T)
        normals.append([v, k])
        b1 = np.linalg.norm(d1 - d2)
        b2 = np.linalg.norm(d1 - d3)
        b3 = np.linalg.norm(d2 - d3)
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
        area_arr.append(area)
        face_jz = np.array([d1, d2, d3])
        tj_arr.append(np.linalg.det(face_jz))
    face_arr = [normals, area_arr, tj_arr]
    vertex_all_id = []
    x_arr1 = []
    x_arr2 = []
    z_arr1 = []
    z_arr2 = []
    for item in range(len(data['vertex'])):
        if data['vertex'][item][0] >= 0:
            x_arr1.append(data['vertex'][item][0])
        else:
            x_arr2.append(data['vertex'][item][0])
        if data['vertex'][item][2] >= 0:
            z_arr1.append(data['vertex'][item][2])
        else:
            z_arr2.append(data['vertex'][item][2])
    x_dis1 = (max(x_arr1) - min(x_arr1)) / dis1
    x_dis2 = (max(x_arr2) - min(x_arr2)) / dis2
    z_dis1 = (max(z_arr1) - min(z_arr1)) / dis3
    z_dis2 = max(z_arr2) - min(z_arr2) / dis4
    point_arr = [max(x_arr1), min(x_arr2), max(z_arr1), min(z_arr2)]
    for item in range(len(data['vertex'])):
        xd = 0
        zd = 0
        if data['vertex'][item][0] >= 0:
            x_dis = point_arr[0] - data['vertex'][item][0]
            if x_dis <= x_dis1:
                xd = 1
        else:
            x_dis = data['vertex'][item][0] - point_arr[1]
            if x_dis <= x_dis2:
                xd = 1
        if data['vertex'][item][2] >= 0:
            z_dis = point_arr[2] - data['vertex'][item][2]
            if z_dis <= z_dis1:
                zd = 1
        else:
            z_dis = data['vertex'][item][2] - point_arr[3]
            if z_dis <= z_dis2:
                zd = 1
        dian_id = []
        result = {}
        f_arr = []
        n1 = 0
        n = 0
        for face, fid in zip(data['face'], range(len(data['face']))):
            dian_id1 = int(face[0].split('/')[0]) - 1
            dian_id2 = int(face[1].split('/')[0]) - 1
            dian_id3 = int(face[2].split('/')[0]) - 1
            if dian_id1 == item:
                if len(f_arr) != 0:
                    for fa in f_arr:
                        if dian_id2 in fa[1] or dian_id3 in fa[1]:
                            angle = np.clip(np.dot(fa[0], normals[fid][0]), -1, 1)
                            if 0 == abs(angle):
                                n1 = n1 + 1
                f_arr.append([normals[fid][0], [dian_id2, dian_id3]])
                dian_id.append(dian_id2)
                dian_id.append(dian_id3)
            elif dian_id2 == item:
                if len(f_arr) != 0:
                    for fa in f_arr:
                        if dian_id1 in fa[1] or dian_id3 in fa[1]:
                            angle = np.clip(np.dot(fa[0], normals[fid][0]), -1, 1)
                            if 0 == abs(angle):
                                n1 = n1 + 1
                f_arr.append([normals[fid][0], [dian_id1, dian_id3]])
                dian_id.append(dian_id1)
                dian_id.append(dian_id3)
            elif dian_id3 == item:
                if len(f_arr) != 0:
                    for fa in f_arr:
                        if dian_id1 in fa[1] or dian_id2 in fa[1]:
                            angle = np.clip(np.dot(fa[0], normals[fid][0]), -1, 1)
                            if 0 == abs(angle):
                                n1 = n1 + 1
                f_arr.append([normals[fid][0], [dian_id1, dian_id2]])
                dian_id.append(dian_id1)
                dian_id.append(dian_id2)
        if (xd == 1 or zd == 1) and n1 >= 3:
            n = 4
        for c in set(dian_id):
            result[c] = dian_id.count(c)
        m = 0
        for key, value in result.items():
            if value >= 3:
                m = 10000
                break
            elif value == 2:
                m = m + 1
        if m == len(result) and n < 3:
            vertex_all_id.append('普通顶点')
        elif m < len(result) and n < 3:
            vertex_all_id.append('边界顶点')
        elif m == 10000 or n > 3:
            vertex_all_id.append('其它顶点')
    # Determine vertex type
    data['类型'] = vertex_all_id
    # print(len(vertex_all_id))
    c = 0
    b = 0
    for i in vertex_all_id:
        if i == '普通顶点':
            c = c + 1
        if i == '其它顶点':
            b = b + 1
    # print(b, c)
    return data, face_arr, length_






