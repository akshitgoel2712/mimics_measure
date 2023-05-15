#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:46:21 2023

@author: akshitgoel
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import warnings
# WARNING INGORED
warnings.simplefilter(action='ignore', category=FutureWarning)

file1 = "SR_005_Whole (2).txt"
file2 = "SR_005_FL (2).txt"
file3 = "SR_005_TL (2).txt"

# Extracting fiducial point from whole aorta file
with open(file1, "r") as file:
    contents = file.read()
    numbers = re.findall(r'(?<=X: )-?\d+(?:\.\d+)?|(?<=Y: )-?\d+(?:\.\d+)?|(?<=Z: )-?\d+(?:\.\d+)?', contents)
    numbers = [float(n) for n in numbers]
    point_coordinates = []
    row = []
    for number in numbers:
        row.append(number)
    point_coordinates = numbers

# Extracting properties from whole aorta file
with open(file1, "r") as file:
    lines = file.readlines()
    numbers = []
    for line in lines:
        if not re.search(r'[a-zA-Z]', line):
            numbers += re.findall(r'-?\d+(?:\.\d+)?', line)
    numbers = [float(n) for n in numbers]
    whole_aorta = []
    row = []
    for number in numbers:
        row.append(number)
        if len(row) == 16:  
            whole_aorta.append(row)
            row = []
    whole_aorta = pd.DataFrame(whole_aorta, columns=["Px", "Py", "Pz", "Tx", "Ty", "Tz", "Nx", "Ny", "Nz", "BNx", "BNy", "BNz","Dfit", "Scf", "Area", "E"])
    # Remove anomalous points from DataFrame
    mean = np.mean(whole_aorta['Dfit'])
    std = np.std(whole_aorta['Dfit'])
    mask = abs(whole_aorta['Dfit'] - mean) > 3*std
    whole_aorta = whole_aorta.drop(whole_aorta[mask].index)
    whole_aorta = whole_aorta.reset_index(drop=True)
    # Reversing order of dataframe
    # whole_aorta = whole_aorta.iloc[::-1]

# Extracting properties from FL file
with open(file2, "r") as file:
    lines = file.readlines()
    numbers = []
    for line in lines:
        if not re.search(r'[a-zA-Z]', line):
            numbers += re.findall(r'-?\d+(?:\.\d+)?', line)
    numbers = [float(n) for n in numbers]
    FL_aorta = []
    row = []
    for number in numbers:
        row.append(number)
        if len(row) == 16:  
            FL_aorta.append(row)
            row = []
    FL_aorta = pd.DataFrame(FL_aorta, columns=["Px", "Py", "Pz", "Tx", "Ty", "Tz", "Nx", "Ny", "Nz", "BNx", "BNy", "BNz","Dfit", "Scf", "Area", "E"])
    # Remove anomalous points from DataFrame
    mean = np.mean(FL_aorta['Dfit'])
    std = np.std(FL_aorta['Dfit'])
    mask = abs(FL_aorta['Dfit'] - mean) > 3*std
    FL_aorta = FL_aorta.drop(FL_aorta[mask].index)
    FL_aorta = FL_aorta.reset_index(drop=True)

# Extracting properties from TL file
with open(file3, "r") as file:
    lines = file.readlines()
    numbers = []
    for line in lines:
        if not re.search(r'[a-zA-Z]', line):
            numbers += re.findall(r'-?\d+(?:\.\d+)?', line)
    numbers = [float(n) for n in numbers]
    TL_aorta = []
    row = []
    for number in numbers:
        row.append(number)
        if len(row) == 16:  
            TL_aorta.append(row)
            row = []
    TL_aorta = pd.DataFrame(TL_aorta, columns=["Px", "Py", "Pz", "Tx", "Ty", "Tz", "Nx", "Ny", "Nz", "BNx", "BNy", "BNz","Dfit", "Scf", "Area", "E"])
    mean = np.mean(TL_aorta['Dfit'])
    std = np.std(TL_aorta['Dfit'])
    mask = abs(TL_aorta['Dfit'] - mean) > 5*std
    TL_aorta = TL_aorta.drop(TL_aorta[mask].index)
    TL_aorta = TL_aorta.reset_index(drop=True)

# Creating the fiducial line
def fiducial_line(whole_aorta, point_coordinates):
    scale_factor = math.sqrt((point_coordinates[0] - whole_aorta.iloc[0, 0])**2 + (point_coordinates[1] - whole_aorta.iloc[0, 1])**2 + (point_coordinates[2] - whole_aorta.iloc[0, 2])**2)/(whole_aorta.iloc[0, 12]/2)
    fiducial_points = []
    for index, row in whole_aorta.iterrows():
        num_rows = whole_aorta.shape[0] - 1
        # new_index = num_rows - index
        new_index = index
        if new_index == 0:
            fiducial_points.append(point_coordinates)
        else:
            # Finding intersection point
            a = new_index - 1
            b = new_index
            v0 = -1*np.array([whole_aorta.iloc[a, 3], whole_aorta.iloc[a, 4], whole_aorta.iloc[a, 5]])
            v1 = -1*np.array([whole_aorta.iloc[b, 3], whole_aorta.iloc[b, 4], whole_aorta.iloc[b, 5]])
            origin = np.array([whole_aorta.iloc[b, 0], whole_aorta.iloc[b, 1], whole_aorta.iloc[b, 2]])
            t1 = (np.dot((origin-fiducial_points[a]), v1)/(np.dot(v0, v1)))
            intersection = fiducial_points[a] + t1*v0
            # Finding point at a certain distance
            direction = intersection - origin
            direction_normalized = direction/np.linalg.norm(direction)
            distance = (whole_aorta.iloc[b,12]/2)*scale_factor
            new_point = origin + (distance * direction_normalized)
            # Adding new point to array
            fiducial_points.append(new_point)
    fiducial_points = pd.DataFrame(fiducial_points, columns=["Px", "Py", "Pz"])
    return fiducial_points

# Finding where a plane drawn on the whole aorta intersects a different lumen
def point_finder(index, whole_aorta, lumen):
    smallest_pos_distance = np.inf
    smallest_pos_point_distance = 20
    smallest_pos_point = None
    smallest_neg_distance = -np.inf
    smallest_neg_point_distance = 20
    smallest_neg_point = None
    normal_vector = -1*np.array([whole_aorta.iloc[index,3], whole_aorta.iloc[index, 4], whole_aorta.iloc[index,5]])
    origin = np.array([whole_aorta.iloc[index, 0], whole_aorta.iloc[index, 1], whole_aorta.iloc[index, 2]])
    d = -np.dot(normal_vector, origin)
    # Iterating through the dataframe lumen to find the closest points above and below the plane
    for index, row in lumen.iterrows():
        point = np.array([row['Px'], row['Py'], row['Pz']])
        distance = (np.dot(normal_vector, point) + d) / np.linalg.norm(normal_vector)
        distance1 = math.sqrt((point[0] - origin[0])**2 + (point[1] - origin[1])**2 + (point[2] - origin[2])**2)
        if distance > 0 and distance < smallest_pos_distance:
            if distance1 < smallest_pos_point_distance:
                smallest_pos_distance = distance
                smallest_pos_point_distance = distance1
                smallest_pos_point = point
        elif distance < 0 and distance > smallest_neg_distance:
            if distance1 < smallest_neg_point_distance:
                smallest_neg_distance = distance
                smallest_neg_point_distance = distance1
                smallest_neg_point = point
    # Finding where a line between the two points intersects the plane unless only one point exists
    if (np.any(smallest_neg_point) == True) and (np.any(smallest_pos_point) == True):
        v = smallest_neg_point - smallest_pos_point
        t = np.dot(normal_vector, (origin - smallest_pos_point)) / np.dot(normal_vector, v)
        new_point = smallest_pos_point + t * v
    else:
        new_point = ["na", "na", "na"]
    return(new_point)

# Finding all the points that match normal planes to the whole aorta
def lumen_points(whole_aorta, lumen):
    lumen_points = []
    for index, row in whole_aorta.iterrows():
        point = point_finder(index, whole_aorta, lumen)
        lumen_points.append(point)
    lumen_points = pd.DataFrame(lumen_points, columns=["Px", "Py", "Pz"])
    lumen_points = lumen_points.iloc[::-1]
    return(lumen_points)

# Finding all helical angles for a given lumen
def helical_angle(whole_aorta, lumen):
    plane_points = lumen_points(whole_aorta, lumen)
    fiducial = fiducial_line(whole_aorta, point_coordinates)
    helical_angles = []
    for index, row in whole_aorta.iterrows():
        A = np.array([row['Px'], row['Py'], row['Pz']])
        B = np.array([fiducial.iloc[index, 0], fiducial.iloc[index, 1], fiducial.iloc[index, 2]])
        C = np.array([plane_points.iloc[index, 0], plane_points.iloc[index, 1], plane_points.iloc[index, 2]])
        arr = ["na", "na", "na"]
        if not np.array_equal(C, arr):
            AB = B - A
            AC = C - A
            dot_product = np.dot(AB, AC)
            cross_product = np.cross(AB, AC)
            angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
            angle_degrees = np.sign(np.dot(cross_product, np.array([0, 0, 1]))) * np.degrees(angle)
            # mag_AB = np.linalg.norm(AB)
            # mag_AC = np.linalg.norm(AC)
            # angle = np.arccos(dot_product / (mag_AB * mag_AC))
            # angle_degrees = np.degrees(angle)
            helical_angles.append(angle_degrees)
    # print(helical_angles)
    return(helical_angles)

def helical_results(lumen):
    helical_data = helical_angle(whole_aorta, lumen)
    change = np.diff(helical_data)
    if np.max(change) > abs(np.min(change)):
        max_change = np.max(change)
    else:
        max_change = np.min(change)
    average_angle = "Avg angle", np.mean(helical_data)
    sd_angle = "Sd angle",np.std(helical_data)
    average_twist = "Avg twist", np.mean(change)
    sd_twist = "Sd twist",np.std(change)
    peak_twist = "Peak twist", max_change
    return(average_angle, sd_angle, average_twist, sd_twist, peak_twist)

# Finding results
def results(aorta):
    max_diameter = "Max Dfit", aorta['Dfit'].max()
    diameter_avg = "Avg Dfit", aorta['Dfit'].mean()
    diameter_sd = "Sd Dfit", aorta['Dfit'].std()
    Scf_avg = "Avg Scf", aorta['Scf'].mean()
    Scf_sd = "Sd Scf", aorta['Scf'].std()
    area_avg = "Avg area", aorta['Area'].mean()
    area_sd = "Sd area", aorta['Area'].std()
    E_avg = "Avg E", aorta['E'].mean()
    E_sd = "Sd E", aorta['E'].std()

    return max_diameter, diameter_avg, diameter_sd, Scf_avg, Scf_sd, area_avg, area_sd, E_avg, E_sd

# Exporting results to CSV file
def csv(aorta, FL, TL):
    helical_data1 = helical_angle(whole_aorta, FL)
    change1 = np.diff(helical_data1)
    if np.max(change1) > abs(np.min(change1)):
        max_change1 = np.max(change1)
    else:
        max_change1 = np.min(change1)
    helical_data2 = helical_angle(whole_aorta, TL)
    change2 = np.diff(helical_data2)
    if np.max(change2) > abs(np.min(change2)):
        max_change2 = np.max(change2)
    else:
        max_change2 = np.min(change2)
    data = {'Whole Max Dfit': aorta['Dfit'].max(),
            'Whole Avg Dfit': aorta['Dfit'].mean(),
            'Whole Sd dfit': aorta['Dfit'].std(),
            'Whole Avg Scf': aorta['Scf'].mean(),
            'Whole Sd Scf': aorta['Scf'].std(),
            'Whole Avg area': aorta['Area'].mean(),
            'Whole Sd area': aorta['Area'].std(),
            'Whole Avg E': aorta['E'].mean(),
            'Whole Sd E': aorta['E'].std(),
            'FL Max Dfit': FL['Dfit'].max(),
            'FL Avg Dfit': FL['Dfit'].mean(),
            'Fl Sd dfit': FL['Dfit'].std(),
            'FL Avg Scf': FL['Scf'].mean(),
            'FL Sd Scf': FL['Scf'].std(),
            'FL Avg area': FL['Area'].mean(),
            'FL Sd area': FL['Area'].std(),
            'FL Avg E': FL['E'].mean(),
            'FL Sd E': FL['E'].std(),
            'FL Avg angle': np.mean(helical_data1),
            'FL SD angle': np.std(helical_data1),
            'FL Avg twist': np.mean(change1),
            'FL Sd twist': np.std(change1),
            'FL peak twist': max_change1,
            'TL Max Dfit': TL['Dfit'].max(),
            'TL Avg Dfit': TL['Dfit'].mean(),
            'Tl Sd dfit': TL['Dfit'].std(),
            'TL Avg Scf': TL['Scf'].mean(),
            'TL Sd Scf': TL['Scf'].std(),
            'TL Avg area': TL['Area'].mean(),
            'TL Sd area': TL['Area'].std(),
            'TL Avg E': TL['E'].mean(),
            'TL Sd E': TL['E'].std(),
            'TL Avg angle': np.mean(helical_data2),
            'TL SD angle': np.std(helical_data2),
            'TL Avg twist': np.mean(change2),
            'TL Sd twist': np.std(change2),
            'TL peak twist': max_change2}
    df = pd.DataFrame(data, index=[0])
    return df

# Plotting 3 graphs on a 3D axis
def plots(dataframe1, dataframe2, dataframe3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = dataframe1.iloc[:, 0]
    y1 = dataframe1.iloc[:, 1]
    z1 = dataframe1.iloc[:, 2]
    x2 = dataframe2.iloc[:, 0]
    y2 = dataframe2.iloc[:, 1]
    z2 = dataframe2.iloc[:, 2]
    x3 = dataframe3.iloc[:, 0]
    y3 = dataframe3.iloc[:, 1]
    z3 = dataframe3.iloc[:, 2]
    
    for index, row in dataframe1.iloc[::5].iterrows():
        ax.scatter(dataframe1.iloc[index, 0], dataframe1.iloc[index, 1], dataframe1.iloc[index, 2], c='k', marker='x')
        ax.scatter(dataframe2.iloc[index, 0], dataframe2.iloc[index, 1], dataframe2.iloc[index, 2], c='g', marker='x')
        ax.scatter(dataframe3.iloc[index, 0], dataframe3.iloc[index, 1], dataframe3.iloc[index, 2], c='b', marker='x')
    ax.plot(x1, y1, z1, c='k', label='whole')
    ax.plot(x2, y2, z2, c='g', label='fiducial')
    ax.plot(x3, y3, z3, c='b', label='TL')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# helical_angles1 = helical_angle(whole_aorta, FL_aorta)
fiducial = fiducial_line(whole_aorta, point_coordinates)
plots(whole_aorta, fiducial, FL_aorta)

df = csv(whole_aorta, FL_aorta, TL_aorta)
df.to_csv('output.csv', index=False)
# print("Whole aorta\n",results(whole_aorta))
# print("FL aorta\n", results(FL_aorta))
# print(helical_results(FL_aorta))
# print("TL aorta\n", results(TL_aorta))
# print(helical_results(TL_aorta))

