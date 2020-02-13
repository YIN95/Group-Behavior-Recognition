import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os

# input_file = csv.reader(open(str(os.getcwd()) + "/test/25.csv"))

# fig = plt.figure()
#
#
# ax = Axes3D(fig)
# n = 100


import numpy as np
import torch


test = list(range(1, 38))
group=[test,test,test,test]
# print(type(group))

def vissingle(index, agentid, weights, timelist):
    input_file = csv.reader(open(str(os.getcwd()) + "/test/Group5/28.csv"))
    ###############################################################
    connections = [['HeadSide', 'HeadFront'],
                   ['HeadFront', 'HeadTop'],
                   ['HeadTop', 'BackTop'],

                   ['BackTop', 'BackLeft'],
                   ['BackLeft', 'LShoulderBack'],
                   ['LShoulderBack', 'LShoulderTop'],
                   ['LShoulderTop', 'Chest'],
                   ['LShoulderBack', 'LElbowOut'],
                   ['LShoulderTop', 'LElbowOut'],
                   ['LElbowOut', 'LUArmHigh'],
                   ['LUArmHigh', 'LHandOut'],
                   ['LHandOut', 'LWristOut'],
                   ['LHandOut', 'LWristIn'],

                   ['BackTop', 'BackRight'],
                   ['BackRight', 'RShoulderBack'],
                   ['RShoulderBack', 'RShoulderTop'],
                   ['RShoulderTop', 'Chest'],
                   ['RShoulderBack', 'RElbowOut'],
                   ['RShoulderTop', 'RElbowOut'],
                   ['RElbowOut', 'RUArmHigh'],
                   ['RUArmHigh', 'RHandOut'],
                   ['RHandOut', 'RWristOut'],
                   ['RWristOut', 'RWristIn'],

                   ['WaistRBack', 'WaistLBack'],

                   ['WaistLFront', 'WaistLBack'],
                   ['WaistLFront', 'LThigh'],
                   ['WaistLBack', 'LKneeOut'],
                   ['LThigh', 'LKneeOut'],
                   ['LKneeOut', 'LShin'],
                   ['LKneeOut', 'LAnkleOut'],
                   ['LShin', 'LAnkleOut'],
                   ['LAnkleOut', 'LToeOut'],
                   ['LToeOut', 'LToeIn'],

                   ['WaistRFront', 'WaistRBack'],
                   ['WaistRFront', 'RThigh'],
                   ['WaistRBack', 'RKneeOut'],
                   ['RThigh', 'RKneeOut'],
                   ['RKneeOut', 'RShin'],
                   ['RKneeOut', 'RAnkleOut'],
                   ['RShin', 'RAnkleOut'],
                   ['RAnkleOut', 'RToeOut'],
                   ['RToeOut', 'RToeIn']
                   ]

    point_labels = ['WaistLFront', 'WaistRFront', 'WaistLBack', 'WaistRBack', 'BackTop', 'Chest', 'BackLeft',
                    'BackRight',
                    'HeadTop', 'HeadFront', 'HeadSide', 'LShoulderBack',
                    'LShoulderTop', 'LElbowOut', 'LUArmHigh', 'LHandOut', 'LWristOut', 'LWristIn', 'RShoulderBack',
                    'RShoulderTop', 'RElbowOut', 'RUArmHigh', 'RHandOut',
                    'RWristOut', 'RWristIn', 'LKneeOut', 'LThigh', 'LAnkleOut', 'LShin', 'LToeOut', 'LToeIn',
                    'RKneeOut',
                    'RThigh', 'RAnkleOut', 'RShin', 'RToeOut', 'RToeIn']

    for i, row in enumerate(input_file):
        if i == 2:
            marker_indexes = [i for i, x in enumerate(row) if x == "Bone Marker"]
            ply = []
            for j in range(4):
                ply.append(marker_indexes[j * 111:(j + 1) * 111])

                # print(len(ply))

        if i == 3:
            marker_labels = [row[i] for i in marker_indexes][::3]
        # if i >= 180 and i%180==0:
        if i in timelist:
            markers = []
            fig = plt.figure()
            # plt.title('frame {}'.format(i))
            # ax = Axes3D(fig)
            ax = fig.gca(projection='3d')
            ax.set_xlim(-1, 1)
            ax.set_zlim(-2, 2)
            c = None
            for m in range(4):
                markers = []
                if m!=agentid:
                    continue
                # fig = plt.figure(m + 1)
                # ax = fig.gca(projection='3d')
                # ax.set_aspect('equal')

                # ax.axis('equal')
                plm = ply[m]
                # plytemp=np.zeros(37)
                # ply1=plytemp
                # for i in marker_indexes:
                for ind in plm:
                    if row[ind] is not '':
                        markers.append(float(row[ind]))
                    else:
                        markers.append(0)
                points = [markers[start::3] for start in range(3)]
                xs, ys, zs = points
                c = ax.scatter(xs, ys, zs, s=200*weights)

                for connection in connections:
                    start, end = connection
                    # start_index = point_labels.index(start) + len(point_labels)
                    # end_index = point_labels.index(end) + len(point_labels)
                    start_index = point_labels.index(start)
                    end_index = point_labels.index(end)
                    if m == 0:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'r-', linewidth=0.8)
                    elif m == 1:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'g-', linewidth=0.8)
                    elif m == 2:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'b-', linewidth=0.8)
                    elif m == 3:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'b-', linewidth=0.8)
                        # ax.axis(False)

            plt.axis('off')
            # fig.colorbar(c, ax=ax)
            ax.view_init(-50,90)
            # ax.view_init(90, -90)
            plt.savefig('attention/model_' + str(index) + '__' + str(i) + 'agentID__'+str(agentid)+'.png')
            plt.close()

def vis(index, group):
    input_file = csv.reader(open(str(os.getcwd()) + "/test/Group5/25.csv"))
###############################################################
    connections = [['HeadSide', 'HeadFront'],
                   ['HeadFront', 'HeadTop'],
                   ['HeadTop', 'BackTop'],

                   ['BackTop', 'BackLeft'],
                   ['BackLeft', 'LShoulderBack'],
                   ['LShoulderBack', 'LShoulderTop'],
                   ['LShoulderTop', 'Chest'],
                   ['LShoulderBack', 'LElbowOut'],
                   ['LShoulderTop', 'LElbowOut'],
                   ['LElbowOut', 'LUArmHigh'],
                   ['LUArmHigh', 'LHandOut'],
                   ['LHandOut', 'LWristOut'],
                   ['LHandOut', 'LWristIn'],

                   ['BackTop', 'BackRight'],
                   ['BackRight', 'RShoulderBack'],
                   ['RShoulderBack', 'RShoulderTop'],
                   ['RShoulderTop', 'Chest'],
                   ['RShoulderBack', 'RElbowOut'],
                   ['RShoulderTop', 'RElbowOut'],
                   ['RElbowOut', 'RUArmHigh'],
                   ['RUArmHigh', 'RHandOut'],
                   ['RHandOut', 'RWristOut'],
                   ['RWristOut', 'RWristIn'],

                   ['WaistRBack', 'WaistLBack'],

                   ['WaistLFront', 'WaistLBack'],
                   ['WaistLFront', 'LThigh'],
                   ['WaistLBack', 'LKneeOut'],
                   ['LThigh', 'LKneeOut'],
                   ['LKneeOut', 'LShin'],
                   ['LKneeOut', 'LAnkleOut'],
                   ['LShin', 'LAnkleOut'],
                   ['LAnkleOut', 'LToeOut'],
                   ['LToeOut', 'LToeIn'],

                   ['WaistRFront', 'WaistRBack'],
                   ['WaistRFront', 'RThigh'],
                   ['WaistRBack', 'RKneeOut'],
                   ['RThigh', 'RKneeOut'],
                   ['RKneeOut', 'RShin'],
                   ['RKneeOut', 'RAnkleOut'],
                   ['RShin', 'RAnkleOut'],
                   ['RAnkleOut', 'RToeOut'],
                   ['RToeOut', 'RToeIn']
                   ]

    point_labels = ['WaistLFront', 'WaistRFront', 'WaistLBack', 'WaistRBack', 'BackTop', 'Chest', 'BackLeft', 'BackRight',
                    'HeadTop', 'HeadFront', 'HeadSide', 'LShoulderBack',
                    'LShoulderTop', 'LElbowOut', 'LUArmHigh', 'LHandOut', 'LWristOut', 'LWristIn', 'RShoulderBack',
                    'RShoulderTop', 'RElbowOut', 'RUArmHigh', 'RHandOut',
                    'RWristOut', 'RWristIn', 'LKneeOut', 'LThigh', 'LAnkleOut', 'LShin', 'LToeOut', 'LToeIn', 'RKneeOut',
                    'RThigh', 'RAnkleOut', 'RShin', 'RToeOut', 'RToeIn']

    for i, row in enumerate(input_file):
        if i == 2:
            marker_indexes = [i for i, x in enumerate(row) if x == "Bone Marker"]
            ply = []
            for j in range(4):
                ply.append(marker_indexes[j * 111:(j + 1) * 111])

            # print(len(ply))

        if i == 3:
            marker_labels = [row[i] for i in marker_indexes][::3]
        # if i >= 180 and i%180==0:
        if i == 720:
            markers = []
            fig=plt.figure()
            # plt.title('frame {}'.format(i))
            # ax = Axes3D(fig)
            ax = fig.gca(projection='3d')
            ax.set_xlim(-1, 1)
            ax.set_zlim(-2, 2)
            c=None
            for m in range(4):
                markers = []
                # fig = plt.figure(m + 1)
                # ax = fig.gca(projection='3d')
                # ax.set_aspect('equal')

                # ax.axis('equal')
                plm = ply[m]
                ply1=group[m]
                # for i in marker_indexes:
                for ind in plm:
                    if row[ind] is not '':
                        markers.append(float(row[ind]))
                    else:
                        markers.append(0)
                points = [markers[start::3] for start in range(3)]
                xs, ys, zs = points
                c=ax.scatter(xs, ys, zs, c=ply1, cmap='bwr')



                for connection in connections:
                    start, end = connection
                    # start_index = point_labels.index(start) + len(point_labels)
                    # end_index = point_labels.index(end) + len(point_labels)
                    start_index = point_labels.index(start)
                    end_index = point_labels.index(end)
                    if m == 0:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'r-', linewidth=0.8)
                    elif m == 1:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'g-', linewidth=0.8)
                    elif m == 2:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'b-', linewidth=0.8)
                    elif m == 3:
                        ax.plot3D([xs[start_index], xs[end_index]], [ys[start_index], ys[end_index]],
                                  [zs[start_index], zs[end_index]], 'y-', linewidth=0.8)
                # ax.axis(False)

            plt.axis('off')
            fig.colorbar(c, ax=ax)
            ax.view_init(-50, 90)
            plt.savefig('attention/model_' + str(index)+'__'+str(i) + '.png')
            plt.close()
            # plt.show()

