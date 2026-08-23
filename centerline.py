#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aortic Dissection Centerline Geometric Pipeline
- Topological segment chaining for complex Mimics exports
- Automatic proximal-to-distal orientation alignment
- Coordinate standardization to origin (0, 0, 0)
- Comparative Helical Twist: Local Polar Projection vs. Iterative Fiducial Guideline
"""
from scipy.signal import savgol_filter
import math
import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress pandas future warnings for cleaner terminal output
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Global Constants & Configuration ---
# Standard column names exported from Mimics centerline data
COL_NAMES = [
    "Px", "Py", "Pz",       # Position coordinates (3D spatial)
    "Tx", "Ty", "Tz",       # Tangent vectors (direction of centerline)
    "Nx", "Ny", "Nz",       # Normal vectors
    "BNx", "BNy", "BNz",    # Binormal vectors
    "Dfit", "Scf", "Area", "E", # Diameter of beset fit, distance of circumference, cross-sectional area, ellipticity
]

EPSILON = 1e-9             # Small value to prevent division by zero in vector math
DISTANCE_TOLERANCE = 25.0  # Search tolerance in mm for mapping lumen points

# --- Batch File Processing ---
def get_file_batches(directory="."):
    """
    Scans a directory and groups centerline text files into processing batches.
    Expects files to be named with a common prefix and a specific suffix 
    (e.g., 'Patient1_Whole.txt', 'Patient1_FL.txt', 'Patient1_TL.txt').
    """
    batches = {}
    pattern = re.compile(r"^(.*?)_(FL|TL|Whole)( \(\d+\))?\.txt$")
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)
            lumen_type = match.group(2)
            suffix = match.group(3) if match.group(3) else ""
            
            batch_id = f"{prefix}{suffix}"
            
            if batch_id not in batches:
                batches[batch_id] = {}
            batches[batch_id][lumen_type] = os.path.join(directory, filename)
            
    # Filter for only complete batches (must have Whole, FL, and TL files)
    complete_batches = {k: v for k, v in batches.items() if all(l in v for l in ['Whole', 'FL', 'TL'])}
    return complete_batches

# --- Robust Data Loading & Topological Segment Chaining ---
def extract_fiducial_point(filename):
    """
    Extracts the initial X, Y, Z spatial coordinates from the text header 
    to use as a proximal anatomical anchor (e.g., Left Subclavian Artery).
    """
    with open(filename, "r") as file:
        contents = file.read()
    coords = re.findall(
        r"(?<=X:\s)-?\d+(?:\.\d+)?|(?<=Y:\s)-?\d+(?:\.\d+)?|(?<=Z:\s)-?\d+(?:\.\d+)?",
        contents,
    )
    if len(coords) < 3:
        coords = re.findall(r"-?\d+\.\d+", contents[:200])
    return [float(c) for c in coords[:3]]

def load_centerline(filename, sigma_cutoff=3.0):
    """
    Parses the text file, chains disjointed segments together based on spatial proximity,
    and removes outliers using standard deviation cutoffs on best-fit diameters.
    """
    segments = {}
    current_seg_id = None
    current_rows = []

    # 1. Parse the text file into segments
    with open(filename, "r") as f:
        for line in f:
            line_str = line.strip()

            seg_match = re.search(r"Branch Segment (\d+):", line_str)
            if seg_match:
                if current_seg_id is not None and current_rows:
                    segments[current_seg_id] = current_rows
                current_seg_id = int(seg_match.group(1))
                current_rows = []
                continue

            tokens = line_str.split()
            if len(tokens) == len(COL_NAMES):
                if tokens[0] == "Px":
                    continue
                parsed = []
                for t in tokens:
                    if t.lower() in ["n/a", "na", "nan", "null"]:
                        parsed.append(np.nan)
                    else:
                        try:
                            parsed.append(float(t))
                        except ValueError:
                            parsed.append(np.nan)
                if len(parsed) == len(COL_NAMES) and not np.isnan(parsed[0]):
                    current_rows.append(parsed)

    if current_seg_id is not None and current_rows:
        segments[current_seg_id] = current_rows

    if not segments:
        return pd.DataFrame(columns=COL_NAMES)

    # 2. Topologically chain the segments based on nearest endpoints
    ordered_segments = []
    available_keys = list(segments.keys())

    current_key = available_keys.pop(0)
    ordered_segments.append(np.array(segments[current_key]))

    while available_keys:
        last_pt = ordered_segments[-1][-1, :3]
        best_key = None
        best_dist = np.inf
        flip_needed = False

        for k in available_keys:
            seg_pts = np.array(segments[k])
            d_start = np.linalg.norm(seg_pts[0, :3] - last_pt)
            d_end = np.linalg.norm(seg_pts[-1, :3] - last_pt)

            if d_start < best_dist:
                best_dist = d_start
                best_key = k
                flip_needed = False
            if d_end < best_dist:
                best_dist = d_end
                best_key = k
                flip_needed = True

        # Break if the next segment is too far away (likely an artifact branch)
        if best_dist > 50.0 or best_key is None:
            break

        next_seg = np.array(segments[best_key])
        if flip_needed:
            next_seg = np.flip(next_seg, axis=0)

        ordered_segments.append(next_seg)
        available_keys.remove(best_key)

    full_data = np.vstack(ordered_segments)
    df = pd.DataFrame(full_data, columns=COL_NAMES)

    # 3. Clean duplicates and interpolate missing diameters
    dup_mask = (df[["Px", "Py", "Pz"]].diff().abs() < 1e-4).all(axis=1)
    df = df[~dup_mask].reset_index(drop=True)

    if df["Dfit"].isna().any():
        df["Dfit"] = df["Dfit"].interpolate(method="linear").bfill().ffill()

    # 4. Remove spatial outliers based on standard deviation
    mean_d = df["Dfit"].mean()
    std_d = df["Dfit"].std()
    if std_d > 0:
        mask = abs(df["Dfit"] - mean_d) <= sigma_cutoff * std_d
        df = df[mask].reset_index(drop=True)

    return df

# --- Anatomical Orientation ---
def orient_proximal_to_distal(df, fiducial_init):
    """
    Ensures the 3D centerline flows from the proximal aorta to the distal bifurcation.
    If the endpoint is closer to the fiducial origin than the start point, the array is flipped.
    """
    if len(df) == 0:
        return df

    init_pt = np.array(fiducial_init, dtype=float)
    pt_start = df.iloc[0][["Px", "Py", "Pz"]].to_numpy(dtype=float)
    pt_end = df.iloc[-1][["Px", "Py", "Pz"]].to_numpy(dtype=float)

    if np.linalg.norm(pt_end - init_pt) < np.linalg.norm(pt_start - init_pt):
        df = df.iloc[::-1].reset_index(drop=True)
        # Reverse vectors accordingly
        df["Tx"] = -df["Tx"]
        df["Ty"] = -df["Ty"]
        df["Tz"] = -df["Tz"]
        df["BNx"] = -df["BNx"]
        df["BNy"] = -df["BNy"]
        df["BNz"] = -df["BNz"]

    return df

# --- Standardization ---
def standardize_coordinates(whole, fl, tl, fiducial_init, scale_by="diameter"):
    """Translates spatial coordinates to a standard origin (0,0,0) and applies scaling."""
    origin = whole.iloc[0][["Px", "Py", "Pz"]].to_numpy(dtype=float)

    if scale_by == "diameter":
        scale_val = float(whole.iloc[0]["Dfit"])
    elif scale_by == "length":
        diffs = whole[["Px", "Py", "Pz"]].diff().dropna().to_numpy()
        scale_val = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        scale_val = 1.0

    if abs(scale_val) < EPSILON:
        scale_val = 1.0

    def transform_df(df):
        df_mod = df.copy()
        df_mod["Px"] = (df_mod["Px"] - origin[0]) / scale_val
        df_mod["Py"] = (df_mod["Py"] - origin[1]) / scale_val
        df_mod["Pz"] = (df_mod["Pz"] - origin[2]) / scale_val
        return df_mod

    whole_std = transform_df(whole)
    fl_std = transform_df(fl)
    tl_std = transform_df(tl)
    fiducial_init_std = (np.array(fiducial_init, dtype=float) - origin) / scale_val

    return whole_std, fl_std, tl_std, fiducial_init_std.tolist(), scale_val

# --- Geometric Calculations ---
def fiducial_line(whole_aorta, point_coordinates, scale_val):
    """
    Constructs a longitudinal reference guideline to decouple luminal twist from whole-vessel bending.
    Iteratively projects a stable reference point forward along the parent vessel's tangent path.
    """
    origin_0 = whole_aorta.iloc[0][["Px", "Py", "Pz"]].to_numpy(dtype=float)
    init_pt = np.array(point_coordinates, dtype=float)

    d_fit_0_spatial = (whole_aorta.iloc[0]["Dfit"] / scale_val) / 2.0
    r0 = d_fit_0_spatial if abs(d_fit_0_spatial) > EPSILON else 1.0
    dist_to_start = np.linalg.norm(init_pt - origin_0)
    
    # Establish baseline normal if fiducial point is out of bounds
    if dist_to_start > 3.0 * r0:
        normal_0 = whole_aorta.iloc[0][["Nx", "Ny", "Nz"]].to_numpy(dtype=float)
        if np.isnan(normal_0).any() or np.linalg.norm(normal_0) < EPSILON:
            t0 = whole_aorta.iloc[0][["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
            ref = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            normal_0 = np.cross(t0, ref)
        normal_0 = normal_0 / np.linalg.norm(normal_0)
        fiducial_points = [origin_0 + r0 * normal_0]
    else:
        fiducial_points = [init_pt]

    scale_factor = np.linalg.norm(fiducial_points[0] - origin_0) / r0

    # Iteratively project guideline along the centerline
    for index in range(1, len(whole_aorta)):
        a = index - 1
        b = index
        v0 = -1.0 * whole_aorta.iloc[a][["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
        v1 = -1.0 * whole_aorta.iloc[b][["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
        origin_b = whole_aorta.iloc[b][["Px", "Py", "Pz"]].to_numpy(dtype=float)

        dot_v0_v1 = np.dot(v0, v1)
        if abs(dot_v0_v1) < EPSILON:
            dot_v0_v1 = EPSILON if dot_v0_v1 >= 0 else -EPSILON

        t1 = np.dot((origin_b - fiducial_points[a]), v1) / dot_v0_v1
        intersection = fiducial_points[a] + t1 * v0

        direction = intersection - origin_b
        norm_dir = np.linalg.norm(direction)
        direction_normalized = direction / (norm_dir + EPSILON) if norm_dir > 0 else np.zeros_like(direction)

        distance = ((whole_aorta.iloc[b]["Dfit"] / scale_val) / 2.0) * scale_factor
        new_point = origin_b + (distance * direction_normalized)
        fiducial_points.append(new_point)

    return pd.DataFrame(fiducial_points, columns=["Px", "Py", "Pz"])

def point_finder(index, whole_aorta, lumen_pts, search_radius):
    """Locates the nearest valid lumen point orthogonally mapped to the whole aorta's centerline frame."""
    normal_vector = -1.0 * whole_aorta.iloc[index][["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
    origin = whole_aorta.iloc[index][["Px", "Py", "Pz"]].to_numpy(dtype=float)

    norm_mag = np.linalg.norm(normal_vector) + EPSILON
    d = -np.dot(normal_vector, origin)

    plane_distances = (np.dot(lumen_pts, normal_vector) + d) / norm_mag
    point_distances = np.linalg.norm(lumen_pts - origin, axis=1)

    pos_mask = (plane_distances > 0) & (point_distances < search_radius)
    neg_mask = (plane_distances < 0) & (point_distances < search_radius)

    # Intersect the lumen with the orthogonal plane
    if np.any(pos_mask) and np.any(neg_mask):
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        best_pos_idx = pos_indices[np.argmin(plane_distances[pos_indices])]
        best_neg_idx = neg_indices[np.argmax(plane_distances[neg_indices])]

        smallest_pos_point = lumen_pts[best_pos_idx]
        smallest_neg_point = lumen_pts[best_neg_idx]

        v = smallest_neg_point - smallest_pos_point
        v_dot_n = np.dot(normal_vector, v)
        if abs(v_dot_n) < EPSILON:
            v_dot_n = EPSILON if v_dot_n >= 0 else -EPSILON

        t = np.dot(normal_vector, (origin - smallest_pos_point)) / v_dot_n
        return smallest_pos_point + t * v
    return [np.nan, np.nan, np.nan]

def lumen_points(whole_aorta, lumen, search_radius):
    """Maps all true/false lumen points along the parent aorta's coordinate frame."""
    lumen_pts = lumen[["Px", "Py", "Pz"]].to_numpy(dtype=float)
    matched_pts = [point_finder(i, whole_aorta, lumen_pts, search_radius=search_radius) for i in range(len(whole_aorta))]
    df = pd.DataFrame(matched_pts, columns=["Px", "Py", "Pz"])
    return df

def calculate_arc_length(df):
    """Calculates the cumulative Euclidean distance (in mm) along the 3D centerline."""
    pts = df[["Px", "Py", "Pz"]].to_numpy(dtype=float)
    distances = np.zeros(len(pts))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return distances

def calculate_tortuosity(df):
    """Calculates Tortuosity Index: total Arc Length divided by direct Chord Length."""
    pts = df[["Px", "Py", "Pz"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return np.nan
    arc_length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    chord_length = np.linalg.norm(pts[-1] - pts[0])
    return arc_length / chord_length if chord_length > EPSILON else np.nan

def calculate_mean_curvature(df, distances_mm):
    """Calculates the average local curvature (kappa) using tangent differentials."""
    T = df[["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
    if len(T) < 2:
        return np.nan
        
    dT = np.linalg.norm(np.diff(T, axis=0), axis=1)
    ds = np.diff(distances_mm)
    
    # Protect against zero distance steps (duplicate points)
    valid_steps = ds > 1e-6
    if not np.any(valid_steps):
        return np.nan
        
    curvature = dT[valid_steps] / ds[valid_steps]
    return np.nanmean(curvature)

def calculate_volume(areas, distances_mm):
    """Calculates Segmental Volumes using 3D Frustum Integration."""
    areas = np.array(areas, dtype=float)
    valid = ~np.isnan(areas) & ~np.isnan(distances_mm)
    areas, distances = areas[valid], distances_mm[valid]
    
    if len(areas) < 2:
        return 0.0
        
    dz = np.diff(distances)
    A1, A2 = areas[:-1], areas[1:]
    
    # Frustum Volume Formula: V = (1/3) * (A1 + A2 + sqrt(A1*A2)) * dz
    volumes = (1.0 / 3.0) * (A1 + A2 + np.sqrt(A1 * A2)) * dz
    return np.sum(volumes)

# METHOD 1: Iterative Fiducial Guideline (Bondesson Method)
def helical_angle_fiducial(whole_aorta, lumen, point_coords, scale_val, search_radius):
    """Computes spatial twist referenced against the stable longitudinal fiducial guideline."""
    plane_points = lumen_points(whole_aorta, lumen, search_radius)
    fiducial = fiducial_line(whole_aorta, point_coords, scale_val)
    helical_angles = []
    
    for i in range(len(whole_aorta)):
        A = whole_aorta.iloc[i][["Px", "Py", "Pz"]].to_numpy(dtype=float)
        B = fiducial.iloc[i][["Px", "Py", "Pz"]].to_numpy(dtype=float)
        C = plane_points.iloc[i][["Px", "Py", "Pz"]].to_numpy(dtype=float)
        T = whole_aorta.iloc[i][["Tx", "Ty", "Tz"]].to_numpy(dtype=float)
        
        if np.isnan(C).any() or np.isnan(T).any():
            helical_angles.append(np.nan)
            continue
            
        AB = B - A
        AC = C - A
        dot_product = np.dot(AB, AC)
        cross_product = np.cross(AB, AC)
        
        angle = np.arctan2(np.dot(T, cross_product), dot_product)
        helical_angles.append(np.degrees(angle))
        
    return np.array(helical_angles)

# METHOD 2: Local 2D Polar Projection
def helical_angle_polar(whole_aorta, lumen, search_radius):
    """Computes spatial twist using standard 2D polar mapping (prone to frame-rotation artifacts)."""
    plane_points = lumen_points(whole_aorta, lumen, search_radius)
    helical_angles = []
    
    for i in range(len(whole_aorta)):
        P = whole_aorta.iloc[i][["Px", "Py", "Pz"]].to_numpy(dtype=float)
        N = whole_aorta.iloc[i][["Nx", "Ny", "Nz"]].to_numpy(dtype=float)
        B = whole_aorta.iloc[i][["BNx", "BNy", "BNz"]].to_numpy(dtype=float)
        C = plane_points.iloc[i][["Px", "Py", "Pz"]].to_numpy(dtype=float)
        
        if np.isnan(C).any() or np.isnan(N).any() or np.isnan(B).any():
            helical_angles.append(np.nan)
            continue
            
        vec = C - P
        x = np.dot(vec, N)
        y = np.dot(vec, B)
        
        angle = np.degrees(np.arctan2(y, x))
        helical_angles.append(angle)
        
    return np.array(helical_angles)

def determine_chirality(max_pos, max_neg, threshold=1.5):
    """Classifies geometric chirality based on extrema of local twist rate."""
    has_right = max_pos >= threshold
    has_left = max_neg <= -threshold
    
    if has_right and has_left:
        return "Mixed-Chiral"
    elif has_right:
        return "Right-Chiral"
    elif has_left:
        return "Left-Chiral"
    else:
        return "Non-Helical"
    
def extract_metrics(angles, distances_mm):
    """
    Extracts twist metrics, addressing missing data gaps and applying 
    a dynamically scaled Savitzky-Golay smoothing filter to handle segmentation voxelation.
    """
    valid_mask = ~np.isnan(angles)
    valid_angles = angles[valid_mask]
    valid_distances = distances_mm[valid_mask]
    
    if len(valid_angles) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Unwrap the phase (fix the 360-degree artificial jumps)
    unwrapped_angles = np.unwrap(valid_angles, period=360)
    
    # NEW: Apply Savitzky-Golay smoothing filter safely
    n_points = len(unwrapped_angles)
    
    # Safely calculate the largest odd integer <= array length (max 11)
    if n_points >= 11:
        window = 11
    else:
        window = n_points if n_points % 2 != 0 else n_points - 1
    
    # S-G requires window size > polyorder.
    if window >= 3:
        poly = 3 if window >= 5 else 1
        smoothed_angles = savgol_filter(unwrapped_angles, window_length=window, polyorder=poly)
    else:
        smoothed_angles = unwrapped_angles
    
    # Calculate physical spatial twist rate using SMOOTHED angles
    angle_change = np.diff(smoothed_angles)
    distance_change = np.diff(valid_distances)
    
    valid_steps = distance_change > 1e-3
    if not np.any(valid_steps):
        return np.mean(unwrapped_angles), np.nan, np.nan, np.nan, np.nan, np.nan
        
    twist_rate = angle_change[valid_steps] / distance_change[valid_steps]
    
    # Extract metrics cleanly using absolute max/min
    peak_twist = twist_rate[np.argmax(np.abs(twist_rate))]
    avg_twist = np.mean(twist_rate)
    sd_twist = np.std(twist_rate)
    
    max_pos_twist = np.max(twist_rate) if np.any(twist_rate > 0) else 0.0
    max_neg_twist = np.min(twist_rate) if np.any(twist_rate < 0) else 0.0
        
    return np.mean(unwrapped_angles), avg_twist, sd_twist, peak_twist, max_pos_twist, max_neg_twist

def csv(batch_id, aorta, FL, TL, point_coords, scale_val, search_radius, whole_raw, fl_raw, tl_raw, make_plots):
    """Aggregates all morphological, geometric, and helical algorithms into a final exportable dataset."""
    max_aortic_diameter = whole_raw["Dfit"].max()
    mean_aortic_diameter = whole_raw["Dfit"].mean()
    
    # 1. Distances (Using RAW, unscaled coordinates for absolute mm)
    whole_dist_mm = calculate_arc_length(whole_raw)
    fl_dist_mm = calculate_arc_length(fl_raw)
    tl_dist_mm = calculate_arc_length(tl_raw)
    
    # 2. Geometric Complexity & Tortuosity
    whole_tort = calculate_tortuosity(whole_raw)
    whole_curve = calculate_mean_curvature(whole_raw, whole_dist_mm)
    
    # 3. Surface Area & Volume Estimation (Frustum Integration)
    fl_vol = calculate_volume(fl_raw["Area"], fl_dist_mm)
    tl_vol = calculate_volume(tl_raw["Area"], tl_dist_mm)
    vol_ratio = tl_vol / fl_vol if fl_vol > EPSILON else np.nan
    area_ratio = tl_raw["Area"].mean() / fl_raw["Area"].mean() if fl_raw["Area"].mean() > EPSILON else np.nan
    
    # 4. Lumen Interaction Metrics (The "Braid" Analysis)
    fl_plane = lumen_points(aorta, FL, search_radius)
    tl_plane = lumen_points(aorta, TL, search_radius)
    
    # Convert standardized centroid coordinates back to absolute mm for distance calculations
    P_whole = aorta[["Px", "Py", "Pz"]].to_numpy(dtype=float) * scale_val
    P_fl = fl_plane[["Px", "Py", "Pz"]].to_numpy(dtype=float) * scale_val
    P_tl = tl_plane[["Px", "Py", "Pz"]].to_numpy(dtype=float) * scale_val
    
    fl_eccentricity = np.nanmean(np.linalg.norm(P_fl - P_whole, axis=1))
    tl_eccentricity = np.nanmean(np.linalg.norm(P_tl - P_whole, axis=1))
    inter_lumen_dist = np.nanmean(np.linalg.norm(P_fl - P_tl, axis=1))

    # 5. Advanced Helical Metrics (Using Polar Method)
    fl_polar = helical_angle_polar(aorta, FL, search_radius)
    tl_polar = helical_angle_polar(aorta, TL, search_radius)

    # 6. Advanced Helical Metrics (Using Fiducial Method)
    fl_fiducial = helical_angle_fiducial(aorta, FL, point_coords, scale_val, search_radius)
    tl_fiducial = helical_angle_fiducial(aorta, TL, point_coords, scale_val, search_radius)
    
    # Only calculate metrics where BOTH lumens physically exist in the slice
    overlap_mask = ~np.isnan(fl_fiducial) & ~np.isnan(tl_fiducial)
    
    # Apply the mask to isolate the true dissection zone (Fiducial)
    fl_fiducial_zone = np.where(overlap_mask, fl_fiducial, np.nan)
    tl_fiducial_zone = np.where(overlap_mask, tl_fiducial, np.nan)

    # Apply the mask to isolate the true dissection zone (Polar)
    fl_polar_zone = np.where(overlap_mask, fl_polar, np.nan)
    tl_polar_zone = np.where(overlap_mask, tl_polar, np.nan)

    # Calculate metrics ONLY within the valid dissection zone (Polar)
    fl_mean_pol, fl_avg_tw_pol, fl_sd_tw_pol, fl_peak_tw_pol, _, _ = extract_metrics(fl_polar_zone, whole_dist_mm)
    tl_mean_pol, tl_avg_tw_pol, tl_sd_tw_pol, tl_peak_tw_pol, _, _ = extract_metrics(tl_polar_zone, whole_dist_mm)
    
    # Calculate metrics ONLY within the valid dissection zone (Fiducial)
    fl_mean_fid, fl_avg_tw_fid, fl_sd_tw_fid, fl_peak_tw_fid, fl_max_pos, fl_max_neg = extract_metrics(fl_fiducial_zone, whole_dist_mm)
    tl_mean_fid, tl_avg_tw_fid, tl_sd_tw_fid, tl_peak_tw_fid, tl_max_pos, tl_max_neg = extract_metrics(tl_fiducial_zone, whole_dist_mm)
    
    # Pitch and Chirality Mapping
    fl_pitch = (360.0 / abs(fl_avg_tw_fid)) if abs(fl_avg_tw_fid) > EPSILON else np.nan
    fl_chirality = determine_chirality(fl_max_pos, fl_max_neg, threshold=0.5)
    
    tl_pitch = (360.0 / abs(tl_avg_tw_fid)) if abs(tl_avg_tw_fid) > EPSILON else np.nan
    tl_chirality = determine_chirality(tl_max_pos, tl_max_neg, threshold=0.5)
    
    # Compile into structured dictionary
    data = {
        # Diameter
        "Max Aortic Diameter (mm)": max_aortic_diameter,
        "Mean Aortic Diameter (mm)": mean_aortic_diameter,
 
        # Global Complexity
        "Tortuosity Index": whole_tort,
        "Mean Curvature (1/mm)": whole_curve,
        
        # Volumetric & Area Interactions
        "TL Volume (mm3)": tl_vol,
        "FL Volume (mm3)": fl_vol,
        "TL/FL Volume Ratio": vol_ratio,
        "TL/FL Mean Area Ratio": area_ratio,
        
        # Spatial Eccentricity
        "Mean Inter-Lumen Dist (mm)": inter_lumen_dist,
        "TL Eccentricity (mm)": tl_eccentricity,
        "FL Eccentricity (mm)": fl_eccentricity,
        
        # False Lumen Helix (Fiducial)
        "FL Avg Twist Fiducial (deg/mm)": fl_avg_tw_fid,
        "FL Twist SD (deg/mm)": fl_sd_tw_fid,
        "FL Peak Twist Fiducial (deg/mm)": fl_peak_tw_fid,
        "FL Spiral Pitch (mm)": fl_pitch,
        "FL Chirality": fl_chirality,
        
        # False Lumen Helix (Polar - For Validation)
        "FL Avg Twist Polar (deg/mm)": fl_avg_tw_pol,
        
        # True Lumen Helix (Fiducial)
        "TL Avg Twist Fiducial (deg/mm)": tl_avg_tw_fid,
        "TL Twist SD (deg/mm)": tl_sd_tw_fid,
        "TL Peak Twist Fiducial (deg/mm)": tl_peak_tw_fid,
        "TL Spiral Pitch (mm)": tl_pitch,
        "TL Chirality": tl_chirality,
        
        # True Lumen Helix (Polar - For Validation)
        "TL Avg Twist Polar (deg/mm)": tl_avg_tw_pol,
    }
    
    if make_plots == True:
        fiducial = fiducial_line(whole_aorta, point_coords, scale_val)
        plots(batch_id, aorta, TL, fiducial, step=5)
    return pd.DataFrame(data, index=[0])

# --- 3D Visualization ---
def plots(batch_id, dataframe1, dataframe2, dataframe3, step=5):
    """Generates a 3D scatter plot of the extracted whole aorta, lumen, and fiducial guideline."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Continuous lines
    ax.plot(dataframe1.iloc[:, 0], dataframe1.iloc[:, 1], dataframe1.iloc[:, 2], c="k", linewidth=2, label="Whole Aorta")
    ax.plot(dataframe2.iloc[:, 0], dataframe2.iloc[:, 1], dataframe2.iloc[:, 2], c="g", linewidth=1.5, linestyle="--", label="Fiducial")
    ax.plot(dataframe3.iloc[:, 0], dataframe3.iloc[:, 1], dataframe3.iloc[:, 2], c="b", linewidth=1.8, label="FL/TL")

    # 2. Subsampled scatter points
    sub1 = dataframe1.iloc[::step]
    sub2 = dataframe2.iloc[::step]
    sub3 = dataframe3.iloc[::step]

    ax.scatter(sub1.iloc[:, 0], sub1.iloc[:, 1], sub1.iloc[:, 2], c="k", marker="x", s=20)
    ax.scatter(sub2.iloc[:, 0], sub2.iloc[:, 1], sub2.iloc[:, 2], c="g", marker="o", s=15)
    ax.scatter(sub3.iloc[:, 0], sub3.iloc[:, 1], sub3.iloc[:, 2], c="b", marker="^", s=20)

    # 3. Enforce true 1:1:1 equal aspect ratio to prevent box distortion
    all_x = np.concatenate([dataframe1.iloc[:, 0], dataframe2.iloc[:, 0], dataframe3.iloc[:, 0]])
    all_y = np.concatenate([dataframe1.iloc[:, 1], dataframe2.iloc[:, 1], dataframe3.iloc[:, 1]])
    all_z = np.concatenate([dataframe1.iloc[:, 2], dataframe2.iloc[:, 2], dataframe3.iloc[:, 2]])

    max_range = np.array([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()
    ])
    
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    radius = 0.5 * max_range.max()

    ax.set_xlim(mid_x - radius, mid_x + radius)
    ax.set_ylim(mid_y - radius, mid_y + radius)
    ax.set_zlim(mid_z - radius, mid_z + radius)

    ax.set_box_aspect((max_range[0] / max_range.max(),
                       max_range[1] / max_range.max(),
                       max_range[2] / max_range.max()))

    ax.set_xlabel("X (norm)")
    ax.set_ylabel("Y (norm)")
    ax.set_zlabel("Z (norm)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    plot_dir = "Plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # Save the figure with the patient ID as the filename
    filepath = os.path.join(plot_dir, f"{batch_id}_plot.png")
    plt.savefig(filepath, dpi=150)
    print(f"Saved plot: {filepath}")
    
    # Close the figure to prevent the memory leak!
    plt.close(fig)

    
# --- Main Execution Pipeline ---
if __name__ == "__main__":
    
    target_directory = "." # Change if your files are in a specific folder, e.g., "./data"
    batches = get_file_batches(target_directory)
    
    if not batches:
        print("No complete file batches found in the directory.")
    else:
        print(f"Found {len(batches)} complete datasets to process.")
        
    all_results = []
    
    for batch_id, files in batches.items():
        print(f"Processing Dataset: {batch_id}...")
        try:
            # 1. Load Data
            point_coordinates_raw = extract_fiducial_point(files['Whole'])
            whole_raw = load_centerline(files['Whole'], sigma_cutoff=3.0)
            FL_raw = load_centerline(files['FL'], sigma_cutoff=3.0)
            TL_raw = load_centerline(files['TL'], sigma_cutoff=5.0)

            # 2. Orient and Standardize
            whole_raw = orient_proximal_to_distal(whole_raw, point_coordinates_raw)
            FL_raw = orient_proximal_to_distal(FL_raw, point_coordinates_raw)
            TL_raw = orient_proximal_to_distal(TL_raw, point_coordinates_raw)

            whole_aorta, FL_aorta, TL_aorta, point_coords, scale_val = standardize_coordinates(
                whole_raw, FL_raw, TL_raw, point_coordinates_raw, scale_by="diameter"
            )

            normalized_search_radius = DISTANCE_TOLERANCE / scale_val
            
            # 3. Compute Metrics
            df_result = csv(
                batch_id,
                whole_aorta,
                FL_aorta,
                TL_aorta,
                point_coords,
                scale_val=scale_val,
                search_radius=normalized_search_radius,
                whole_raw=whole_raw,
                fl_raw=FL_raw,       
                tl_raw=TL_raw,
                make_plots=True
            )
            
            # Insert ID for tracking in R
            df_result.insert(0, "Patient_ID", batch_id)
            all_results.append(df_result)
            
        except Exception as e:
            print(f"  -> Error processing {batch_id}: {e}")
            
    # 4. Compile and Export
    if all_results:
        final_dataframe = pd.concat(all_results, ignore_index=True)
        final_dataframe.to_csv("batch_output.csv", index=False)
        print(f"\nBatch complete. Successfully processed {len(all_results)} datasets. Saved to 'batch_output.csv'")
