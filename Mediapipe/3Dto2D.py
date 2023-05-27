import numpy as np
import csv
import os
import cv2
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='compare mediapipe and kinect skeleton data', epilog = 'Enjoy the program! :)')

parser.add_argument('--inputfile', default =r'/datasets_nas/elma1294/example_tape/dominikw_front_0/e4sm_tape-2021-08-18_14-07-44.211275_skeletons.shuttleFront.csv',
                     help = 'The address of the input csv file')
parser.add_argument('--sample_image', default = r'/datasets_nas/elma1294/example_tape/dominikw_front_0/e4sm_tape-2021-08-18_14-07-44.211275_png.shuttleFront/shuttleFront_Kinect_ColorImage1629288474191866000.jpg',
                   help = 'sample image to get the width and height of the image')
parser.add_argument('--output_directory', default=r'/datasets_nas/elma1294/results' , help = 'The address of the results')
arguments = parser.parse_args()

path = arguments.inputfile
file_path = arguments.sample_image
output_directory = arguments.output_directory

output_kineckt = os.path.join(output_directory, 'results3Dto2Dnew.csv')


def _parse_camera_parameter(camera_name: str):

    assert camera_name in ['shuttleFront', 'shuttleSide', 'spike']

    # parse serialized kinect azure calibration
    root = ET.parse(f'intrinsics/{camera_name}.xml').getroot()
    root = root.find('KinectAzureCalibration')

    # get color intrinsics as cx, cy, fx, fy
    color_params = root.find('color_camera_calibration')

    # intrsinics
    intrinsics = color_params.find('intrinsics').find('parameters').findall('item')

    cx = float(intrinsics[0].text)
    cy = float(intrinsics[1].text)
    fx = float(intrinsics[2].text)
    fy = float(intrinsics[3].text)

    # get transformation from depth frame to color frame
    # the kinect sdk defines the local transformation tree as
    # mount frame -> depth_frame -> color_frame
    # so we can just parse the extrinsics from color camera which represents
    # the transformation depth_frame -> color_frame
    extrinsics = color_params.find('extrinsics')

    # NOTE: rotation matrix is serialized as simple vector
    rotation = extrinsics.find('rotation').findall('item')
    translation = extrinsics.find('translation').findall('item')

    rotation_vec = np.zeros(9)
    for i, el in enumerate(rotation):
        rotation_vec[i] = el.text

    translation_vec = np.zeros(3)
    for i, el in enumerate(translation):
        translation_vec[i] = el.text

    # translation is given in mm
    # as we work in m we convert
    translation_vec *= 0.001

    transform = np.eye(4)
    transform[:3, :3] = rotation_vec.reshape(3, 3)
    transform[:3, 3] = translation_vec

    return [cx, cy, fx, fy], transform

def convert_3d_to_2d_position(x,y,z, transform, camera_parameter, width, height):
    ''' with Intrinsic Parameter i defined Camera Matrix to get the 2D Position'''
    
    xc = float(x)
    yc = float(y)
    zc = float(z)
    xc, yc, zc = np.dot(transform, np.array([xc, yc, zc, 1]))[:3]

    #shutlefront
    cx, cy, fx, fy = camera_parameter

    if len(x) < 1 or len(y) < 1 or len(z) <1:
        return 0,0
    k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    one = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    camera_matrix = np.array([[xc],[yc],[zc],[1]])
    m1 = np.dot(k,one)
    output_matrix = np.dot(m1,camera_matrix)
    u_array = (output_matrix[0]/output_matrix[2])/width
    v_array= (output_matrix[1]/output_matrix[2])/height
    u = u_array[0].astype(float)
    v = v_array[0].astype(float)
    return u,v


def _plot_sample(joint_dict, img):
    joints = np.zeros(((len(joint_dict) -1) // 2, 2))
    for ei, joint in enumerate(list(joint_dict.values())[1:]):
        joints[ei // 2, ei % 2] = joint
    plt.imshow(img)
    joints *= img.shape[:2][::-1]
    plt.scatter(joints[:, 0], joints[:, 1], s=10, c='r')
    plt.show()
    # plt.savefig('test.png')


list_of_results = []
camera_params, transform = _parse_camera_parameter('shuttleFront')
img = cv2.imread(file_path, cv2.IMREAD_COLOR)
width = img.shape[1]
height = img.shape[0]

with open(path) as f:
    ''' read the csv file and convert the 3D Position to 2D Position and save it in a new csv file'''
    rows = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    for row in rows:
        mDict = {}
        mDict['timestamp'] = row['timestamp']
        for x_header in row.keys():
            if '.x' in x_header:
                y_header = x_header.replace('.x','.y')
                z_header = x_header.replace('.x','.z')
                x_value = row[x_header]
                y_value = row[y_header]
                z_value = row[z_header]
                u,v= convert_3d_to_2d_position(x_value,y_value,z_value, transform, camera_params, width, height)
                mDict[x_header] = u
                mDict[y_header] = v
        list_of_results.append(mDict)


_plot_sample(mDict, img)

with open(output_kineckt, 'w', encoding='UTF8', newline='') as f2:
    writer = csv.DictWriter(f2, mDict.keys())
    writer.writeheader()
    writer.writerows(list_of_results)

