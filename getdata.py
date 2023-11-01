# Development time：2023/3/30  20:05
def get_obj(url):
    file_obj = open(url, 'r')
    obj_list = file_obj.readlines()
    coordinate_arr = []
    face_arr = []
    # Iterate to get data
    for item in obj_list:
        if item[0: 2] == 'v ':
            arr = [float(item.split()[1]), float(item.split()[2]), float(item.split()[3])]
            coordinate_arr.append(arr)
        elif item[0] == 'f':
            arr = [item.split()[1], item.split()[2], item.split()[3]]
            face_arr.append(arr)
    d_all = {"vertex": coordinate_arr, "face": face_arr}
    return d_all









