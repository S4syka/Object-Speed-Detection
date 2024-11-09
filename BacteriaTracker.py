import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import DBSCAN

# Correct the video path
cap = cv2.VideoCapture('Videos of Bacterias/BadBacterias.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Set to your video's actual frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_time_ms = int(1000 / fps)

# Set up the VideoWriter to save the processed video
output_filename = 'Processed_Frame_Video_with_DBSCAN.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

bacteria = {}
next_bacteria_id = 0
max_distance = 30  # Used for DBSCAN as the maximum distance between points
max_disappeared = 5
min_lifetime = 0  # Increased lifetime threshold
min_area = 50  # Minimum area to ignore smaller objects
max_area = 7000  # Maximum area to consider an object as significant

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

current_frame = 0  # Initialize frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1  # Increment the frame counter

    # Calculate the percentage of video processed
    progress_percentage = (current_frame / frame_count) * 100

    # Make a copy of the original frame for display
    original_frame = frame.copy()

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast
    gray = clahe.apply(gray)

    # Combine CLAHE and background subtraction
    combined = cv2.bitwise_and(gray, fgmask)

    # Apply morphological operations to reduce noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply thresholding
    _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply connected components analysis
    num_labels, labels_im = cv2.connectedComponents(thresh)

    centroids = []
    for label in range(1, num_labels):  # Skip the background label 0
        # Create a mask for each component
        component_mask = (labels_im == label).astype("uint8") * 255

        # Calculate the area of the component
        area = cv2.countNonZero(component_mask)
        # Filter out small objects based on area
        if area < min_area or area > max_area:
            continue

        # Compute moments
        M = cv2.moments(component_mask)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append([cX, cY])

    # Use DBSCAN to cluster the centroids
    if len(centroids) > 0:
        centroids_array = np.array(centroids)
        db = DBSCAN(eps=max_distance, min_samples=1).fit(centroids_array)
        labels = db.labels_

        # Filter out noise points (label = -1)
        unique_labels = set(labels)
        filtered_centroids = []

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            # Get the centroids belonging to the current cluster
            cluster_points = centroids_array[labels == label]
            # Calculate the average centroid of the cluster
            avg_centroid = np.mean(cluster_points, axis=0).astype(int)
            filtered_centroids.append((avg_centroid[0], avg_centroid[1]))

        # Use filtered_centroids for further processing
        centroids = filtered_centroids

    # Proceed with tracking logic using the filtered centroids
    if not bacteria:
        for centroid in centroids:
            bacteria[next_bacteria_id] = {
                'positions': [centroid],
                'disappeared': 0,
                'speed': 0,
                'lifetime': 1,        # Initialize lifetime
                'total_distance': 0   # Initialize total distance moved
            }
            next_bacteria_id += 1
    else:
        if len(centroids) == 0:
            # Increment disappeared count for all tracked bacteria
            for bacteria_id in list(bacteria.keys()):
                bacteria[bacteria_id]['disappeared'] += 1
                if bacteria[bacteria_id]['disappeared'] > max_disappeared:
                    del bacteria[bacteria_id]
            continue  # Skip to the next frame

        bacteria_ids = list(bacteria.keys())
        previous_centroids = [
            bacteria[b_id]['positions'][-1] for b_id in bacteria_ids
        ]

        current_centroids = centroids

        if len(previous_centroids) == 0:
            # Register all centroids as new bacteria
            for centroid in centroids:
                bacteria[next_bacteria_id] = {
                    'positions': [centroid],
                    'disappeared': 0,
                    'speed': 0,
                    'lifetime': 1,
                    'total_distance': 0
                }
                next_bacteria_id += 1
            continue  # Proceed to the next frame

        # Compute distance matrix
        D = dist.cdist(np.array(previous_centroids), np.array(current_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row][col] > max_distance:
                continue

            bacteria_id = bacteria_ids[row]
            bacteria[bacteria_id]['positions'].append(current_centroids[col])
            bacteria[bacteria_id]['disappeared'] = 0

            # Increment lifetime
            bacteria[bacteria_id]['lifetime'] += 1

            # Calculate speed and total distance
            positions = bacteria[bacteria_id]['positions']
            if len(positions) >= 2:
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
                distance_moved = np.sqrt(dx ** 2 + dy ** 2)
                bacteria[bacteria_id]['total_distance'] += distance_moved
                # Instantaneous speed
                speed = distance_moved * fps
                bacteria[bacteria_id]['speed'] = speed
            else:
                bacteria[bacteria_id]['speed'] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle disappeared bacteria
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            bacteria_id = bacteria_ids[row]
            bacteria[bacteria_id]['disappeared'] += 1
            if bacteria[bacteria_id]['disappeared'] > max_disappeared:
                del bacteria[bacteria_id]

        # Register new bacteria for unmatched centroids
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            bacteria[next_bacteria_id] = {
                'positions': [current_centroids[col]],
                'disappeared': 0,
                'speed': 0,
                'lifetime': 1,
                'total_distance': 0
            }
            next_bacteria_id += 1

    num_bacteria = len(bacteria)

    # Variables to calculate average speed of all bacteria
    total_average_speed = 0
    count_bacteria_with_min_lifetime = 0

    # Identify the fastest bacterium among those that have exceeded min_lifetime
    max_speed = -1
    fastest_bacteria_id = None
    for bacteria_id, data in bacteria.items():
        lifetime = data['lifetime']
        if lifetime >= min_lifetime:
            total_time = (lifetime - 1) / fps  # Total time in seconds
            if total_time > 0:
                # Calculate average speed of the bacterium
                average_speed = (data['total_distance'] / total_time)
                data['average_speed'] = average_speed  # Store average speed
            else:
                data['average_speed'] = 0

            # Accumulate total average speed
            total_average_speed += data['average_speed']
            count_bacteria_with_min_lifetime += 1

            # Find the fastest bacterium based on average speed
            if data['average_speed'] > max_speed:
                max_speed = data['average_speed']
                fastest_bacteria_id = bacteria_id

    # Calculate overall average speed
    if count_bacteria_with_min_lifetime > 0:
        overall_average_speed = total_average_speed / count_bacteria_with_min_lifetime
    else:
        overall_average_speed = 0

    # Display the number of bacteria and average speed on the 'Frame' window
    cv2.putText(
        frame,
        f"Bacteria Count: {num_bacteria}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Avg Speed: {overall_average_speed:.2f} px/s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Display speed and highlight the fastest bacterium
    for bacteria_id, data in bacteria.items():
        positions = data['positions']
        lifetime = data['lifetime']
        is_fastest = (bacteria_id == fastest_bacteria_id)

        # Only display bacteria that have a lifetime exceeding min_lifetime
        if lifetime >= min_lifetime:
            average_speed = data['average_speed']

            # Set color for the fastest bacterium
            if is_fastest:
                color = (0, 0, 150)  # Darker red color
                text_color = (0, 0, 150)
            else:
                color = (0, 150, 150)  # Darker yellow color
                text_color = (0, 150, 0)  # Darker green

            # Display average speed
            cv2.putText(
                frame,
                f"Avg Speed: {average_speed:.2f} px/s",
                (positions[-1][0] + 10, positions[-1][1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,  # Font scale
                text_color,
                1     # Thickness
            )

            # Draw centroid on the 'Frame' window
            cv2.circle(frame, positions[-1], 5, color, -1)

            # Also draw on the 'Threshold' window
            cv2.circle(thresh, positions[-1], 5, color, -1)

    # Display progress on the 'Frame' window
    cv2.putText(
        frame,
        f"Progress: {progress_percentage:.2f}%",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Save the processed frame to the video file
    out.write(frame)

    # Display the original video frame
    # cv2.imshow('Original Video', original_frame)

    # Display the frames
    cv2.imshow('Frame', frame)
    # cv2.imshow('Threshold', thresh)  # Show the thresholded image with average speed

    # Speed up the video by reducing the wait time
    key = cv2.waitKey(frame_time_ms // 2)  # Faster playback
    if key == 27:  # Press 'ESC' to exit
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_filename}")
