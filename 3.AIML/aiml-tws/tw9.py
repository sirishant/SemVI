# Description: K-means clustering algorithm implementation
import csv
from math import sqrt
import random
from collections import defaultdict

def euclid(p1, p2): # Euclidean distance between two points
    return sqrt(sum((p1[dim] - p2[dim])**2 for dim in range(len(p1))))  # sqrt((x1-x2)^2 + (y1-y2)^2 + ...)

def initialize_centroids(points, k):
    return random.sample(points, k)     # Randomly select k points from the dataset

def closest_centroid(point, centroids):
    return min(centroids, key=lambda c: euclid(point, c))   # Find the centroid with minimum distance to the point

def assign_clusters(points, centroids): # Assign points to the closest centroid
    clusters = defaultdict(list)    # Dictionary to store points for each cluster
    for point in points:            # Iterate over each point
        closest = closest_centroid(point, centroids)    # Find the closest centroid
        clusters[tuple(closest)].append(point)          # Add the point to the cluster of the closest centroid
    return clusters

def update_centroids(clusters):     # Update the centroids based on the points in each cluster
    new_centroids = []              # List to store the new centroids
    for points in clusters.values():# Iterate over the points in each cluster
        if points:                  # Check if the cluster is not empty
            new_centroid = [sum(p[dim] for p in points) / len(points) for dim in range(len(points[0]))] # Compute the new centroid
            new_centroids.append(new_centroid)  # Add the new centroid to the list
    return new_centroids

def has_converged(old, new, tol=1e-4):  # Check if the centroids have converged
    # Check if the Euclidean distance between the old and new centroids is less than the tolerance
    return all(euclid(old[i], new[i]) <= tol for i in range(len(old)))

def k_means(filename, k, max_iterations=5, tol=1e-4): # tol is the tolerance for convergence
    points = [] # List to store the points from the dataset
    with open(filename, 'r') as file:   # Read the dataset from the file
        reader = csv.reader(file)       # Create a CSV reader
        for row in reader:              # Iterate over each row
            points.append([float(value) for value in row])  # Convert the row values to floats and add to the points list

    centroids = initialize_centroids(points, k) # Initialize the centroids

    for _ in range(max_iterations): # Iterate for a maximum number of iterations
        clusters = assign_clusters(points, centroids)   # Assign points to clusters
        new_centroids = update_centroids(clusters)      # Update the centroids

        if has_converged(centroids, new_centroids, tol):    # Check for convergence
            break       # Break if the centroids have converged
        centroids = new_centroids   # Update the centroids
    return centroids, clusters

def print_d(centroids, clusters):   # Print the clusters and centroids
    print('Centroids:', centroids)
    for centroid, cluster_points in clusters.items():   # Iterate over the clusters
        print(f"Cluster around centroid {centroid}: ")  # Print the cluster points around the centroid
        for p in cluster_points:
            print(p)
        print('-----------------------')

# Example usage
centroids, clusters = k_means('dataset.csv', 2)
print("************************\n")
print_d(centroids, clusters)