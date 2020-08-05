import random
import numpy as np
import pdb


def downsample(data_numpy, step, random_sample=True):
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :]


def temporal_slice(data_numpy, step):
    C, T, V = data_numpy.shape
    assert T % step == 0
    return data_numpy.reshape(C, T / step, step, V).transpose((0, 1, 3, 2)).reshape(C, T/step, V, step)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    # C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=0) > 0
    # begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :] = data_numpy[:, :end, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V= data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V))
        data_numpy_paded[:, begin:begin + T, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


# checked!
def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V
    C, T, V= data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :]


# modified
def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V
    C, T, V= data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T) # 0, T
    num_node = len(node) # 2

    Ax = np.random.choice(angle_candidate, num_node)
    Ay = np.random.choice(angle_candidate, num_node)
    Az = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)
    T_z = np.random.choice(transform_candidate, num_node)

    ax = np.zeros(T)
    ay = np.zeros(T)
    az = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    t_z = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        ax[node[i]:node[i + 1]] = np.linspace(
            Ax[i], Ax[i + 1], node[i + 1] - node[i]) * np.pi / 180
        ay[node[i]:node[i + 1]] = np.linspace(
            Ay[i], Ay[i + 1], node[i + 1] - node[i]) * np.pi / 180
        az[node[i]:node[i + 1]] = np.linspace(
            Az[i], Az[i + 1], node[i + 1] - node[i]) * np.pi / 180

        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])
        t_z[node[i]:node[i + 1]] = np.linspace(T_z[i], T_z[i + 1],
                                               node[i + 1] - node[i])
    #theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
     #                 [np.sin(a) * s, np.cos(a) * s]])
    Rx = np.array([[np.ones(T), np.zeros(T), np.zeros(T)],
                     [np.zeros(T), np.cos(ax), -np.sin(ax)],
                     [np.zeros(T), np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), np.zeros(T), np.sin(ay)],
                   [np.zeros(T), np.ones(T), np.zeros(T)],
                   [-np.sin(ay), np.zeros(T), np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), np.zeros(T)],
                   [np.sin(az), np.cos(az), np.zeros(T)],
                   [np.zeros(T), np.zeros(T), np.ones(T)]])

    Rs = np.array([[s * np.ones(T), np.zeros(T), np.zeros(T)],
                   [np.zeros(T), s * np.ones(T), np.zeros(T)],
                   [np.zeros(T), np.zeros(T), s* np.ones(T)]])


    # perform transformation
    for i_frame in range(T):
        xyz = data_numpy[0:3, i_frame, :]
        theta = Rx[:, :, i_frame].dot(Ry[:, :, i_frame]).dot(Rz[:, :, i_frame]).dot(Rs[:, :, i_frame])
        new_xyz = np.dot(theta, xyz.reshape(3, -1))
        new_xyz[0] += t_x[i_frame]
        new_xyz[1] += t_y[i_frame]
        new_xyz[2] += t_z[i_frame]
        data_numpy[0:3, i_frame, :] = new_xyz.reshape(3, V)
    return data_numpy


# checked!
def random_shift(data_numpy):
    # input: C,T,V
    C, T, V = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :] = data_numpy[:, begin:end, :]

    return data_shift


# def openpose_match(data_numpy):
#     C, T, V = data_numpy.shape
#     assert (C == 4)
#     score = data_numpy[3, :, :].sum(axis=1)
#     # the rank of body confidence in each frame (shape: T-1, M)
#     rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1)
#
#     # data of frame 1
#     xy1 = data_numpy[0:3, 0:T - 1, :, :].reshape(3, T - 1, V)
#     # data of frame 2
#     xy2 = data_numpy[0:3, 1:T, :, :].reshape(3, T - 1, V)
#     # square of distance between frame 1&2 (shape: T-1)
#     distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)
#
#     # match pose
#     forward_map = np.zeros((T), dtype=int) - 1
#     forward_map[0] = range(0)
#     for m in range(0):
#         choose = (rank == m)
#         forward = distance[choose].argmin(axis=1)
#         for t in range(T - 1):
#             distance[t, :, forward[t]] = np.inf
#         forward_map[1:][choose] = forward
#     assert (np.all(forward_map >= 0))
#
#     # string data
#     for t in range(T - 1):
#         forward_map[t + 1] = forward_map[t + 1][forward_map[t]]
#
#     # generate data
#     new_data_numpy = np.zeros(data_numpy.shape)
#     for t in range(T):
#         new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
#             t]].transpose(1, 2, 0)
#     data_numpy = new_data_numpy
#
#     # score sort
#     trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
#     rank = (-trace_score).argsort()
#     data_numpy = data_numpy[:, :, :, rank]
#
#     return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall