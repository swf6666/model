#Development time：2023/4/23  19:49
def new_calculate(data, face_zhi, np, lg_):
    id_all = []
    n = 0
    for face, fid in zip(data['face'], range(len(data['face']))):
        if n == lg_:
            break
        #  Updates triangle information related to collapsed triangles
        if face[3] == 0:
            n = n + 1
            id1 = int(face[0].split('/')[0]) - 1
            id2 = int(face[1].split('/')[0]) - 1
            id3 = int(face[2].split('/')[0]) - 1
            id_all.append([id1, id2, id3])
            d1 = np.array(data['vertex'][id1])
            d2 = np.array(data['vertex'][id2])
            d3 = np.array(data['vertex'][id3])
            v1 = d1 - d2
            v2 = d1 - d3
            v3 = np.cross(v1, v2)
            if v3[1] < 0:
                v3 = -v3
            length = np.sqrt(v3.dot(v3))
            if length == 0:
                length = np.sqrt(v1.dot(v1))
                v = v1 / length
                x = 1 / 3 * (d1[0] + d2[0] + d3[0])
                y = 1 / 3 * (d1[1] + d2[1] + d3[1])
                z = 1 / 3 * (d1[2] + d2[2] + d3[2])
                d = -(x * v[0] + y * v[1] + z * v[2])
                p = np.array([[v[0]], [v[1]], [v[2]], [d]])
                k = np.matmul(p, p.T)
                face_zhi[0][fid] = [v, k]
            else:
                v = v3 / length
                x = 1 / 3 * (d1[0] + d2[0] + d3[0])
                y = 1 / 3 * (d1[1] + d2[1] + d3[1])
                z = 1 / 3 * (d1[2] + d2[2] + d3[2])
                d = -(x * v[0] + y * v[1] + z * v[2])
                p = np.array([[v[0]], [v[1]], [v[2]], [d]])
                k = np.matmul(p, p.T)
                face_zhi[0][fid] = [v, k]
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
            face_zhi[1][fid] = area
            face_jz = np.array([d1, d2, d3])
            face_zhi[2][fid] = np.linalg.det(face_jz)
    return face_zhi
