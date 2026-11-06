"""Constantes de topologie et dimensions pour les repères MediaPipe LSFB."""

# 23 repères supérieurs MediaPipe Pose (nez, yeux, oreilles, bouche, épaules, coudes, poignets, mains)
UPPER_POSE_LANDMARK_COUNT = 23
HAND_LANDMARK_COUNT = 21

# Indices MediaPipe Pose pour le centrage et la normalisation d'échelle
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12

TOTAL_LANDMARKS = UPPER_POSE_LANDMARK_COUNT + (2 * HAND_LANDMARK_COUNT)  # 65 points
COORDINATE_DIM = 3  # (x, y, z)
FEATURE_DIM = TOTAL_LANDMARKS * COORDINATE_DIM  # 195 features par frame

TARGET_SEQUENCE_LENGTH = 40  # Frames par séquence standardisée
EPSILON_DISTANCE = 1e-6

