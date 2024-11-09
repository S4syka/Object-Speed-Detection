
### **Code with Explanations**

```python
import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import DBSCAN

# Correct the video path
cap = cv2.VideoCapture('Videos of Bacterias/GoodBacterias.mp4')
```

- **Imports**: OpenCV (`cv2`) for video and image processing, `numpy` for handling numerical operations, `distance` from `scipy.spatial` for calculating Euclidean distances, and `DBSCAN` for clustering detected bacteria.

---

```python
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Set to your video's actual frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_time_ms = int(1000 / fps)
```

- **Video Properties**: Gets the FPS, width, height, and total frame count of the video. Calculates the time delay between frames for playback.

---

```python
# Set up the VideoWriter to save the processed video
output_filename = 'Processed_Frame_Video_with_DBSCAN.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
```

- **Video Writer Setup**: Prepares the `VideoWriter` to save the processed video using the specified codec (`XVID`) and video properties.

---

```python
bacteria = {}
next_bacteria_id = 0
max_distance = 20  # Used for DBSCAN as the maximum distance between points
max_disappeared = 5
min_lifetime = 5  # Increased lifetime threshold
min_area = 300  # Minimum area to ignore smaller objects
max_area = 7000  # Maximum area to consider an object as significant
```

- **Tracking Parameters**: Defines dictionaries and variables for tracking bacteria, including maximum distance for clustering, disappearance threshold, and area filters.

---

```python
# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

- **Background Subtractor**: Uses MOG2 to isolate moving objects. `clahe` is used to improve contrast in the grayscale images.

---

```python
current_frame = 0  # Initialize frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break
```

- **Frame Counter and Loop**: Initializes the frame counter and starts a loop to read frames from the video. Breaks the loop if no frame is returned.

---

```python
current_frame += 1  # Increment the frame counter

# Calculate the percentage of video processed
progress_percentage = (current_frame / frame_count) * 100

# Make a copy of the original frame for display
original_frame = frame.copy()

# Apply background subtraction
fgmask = fgbg.apply(frame)
```

- **Frame Processing**: Updates the frame counter, calculates progress, copies the frame for display, and applies background subtraction.

---

```python
# Convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply CLAHE to enhance contrast
gray = clahe.apply(gray)

# Combine CLAHE and background subtraction
combined = cv2.bitwise_and(gray, fgmask)
```

- **Grayscale and Contrast Enhancement**: Converts the frame to grayscale, enhances contrast, and combines it with the background mask to isolate moving bacteria.

---

```python
# Apply morphological operations to reduce noise and fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

# Apply thresholding
_, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

- **Noise Reduction and Thresholding**: Uses morphological operations to remove noise and applies Otsu's thresholding to create a binary image.

---

```python
# Apply connected components analysis
num_labels, labels_im = cv2.connectedComponents(thresh)

centroids = []
```

- **Connected Components Analysis**: Labels connected components in the binary image and prepares to store their centroids.

---

```python
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
```

- **Extracting and Filtering Centroids**: Iterates through labels, filters objects by area, and calculates centroids using image moments.

---

```python
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
```

- **DBSCAN Clustering**: Clusters the centroids using DBSCAN and filters out noise points. Calculates the average centroid for each cluster.

---

```python
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
```

- **Initialize Tracking**: If no bacteria are currently tracked, registers new bacteria using the centroids.

---

```python
else:
    if len(centroids) == 0:
        # Increment disappeared count for all tracked bacteria
        for bacteria_id in list(bacteria.keys()):
            bacteria[bacteria_id]['disappeared'] += 1
            if bacteria[bacteria_id]['disappeared'] > max_disappeared:
                del bacteria[bacteria_id]
        continue  # Skip to the next frame
```

- **Handle Missing Centroids**: Increments the disappearance count for bacteria and removes those that have disappeared for too long.

---

```python
bacteria_ids = list(bacteria.keys())
previous_centroids = [bacteria[b_id]['positions'][-1] for b_id in bacteria_ids]
current_centroids = centroids
```

- **Prepare for Matching**: Extracts the previous and current centroids for tracking.

---

```python
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
```

- **Register New Bacteria**: If there are no previous centroids, registers all current centroids as new bacteria.

---

```python
# Compute distance matrix
D = dist.cdist(np.array(previous_centroids), np.array(current_centroids))
rows = D.min(axis=1).argsort()
cols = D.argmin(axis=1)[rows]
```

- **Distance Matrix**: Computes the pairwise distances between previous and current centroids and sorts them for matching.

---

```python
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
    bacteria[bacteria_id]['lifetime'] += 1

    # Calculate speed and total distance
    positions =

 bacteria[bacteria_id]['positions']
    if len(positions) >= 2:
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        distance_moved = np.sqrt(dx ** 2 + dy ** 2)
        bacteria[bacteria_id]['total_distance'] += distance_moved
        speed = distance_moved * fps
        bacteria[bacteria_id]['speed'] = speed
    else:
        bacteria[bacteria_id]['speed'] = 0

    used_rows.add(row)
    used_cols.add(col)
```

- **Match and Update**: Matches centroids, updates positions and lifetime, and calculates speed and distance.

---

```python
# Handle disappeared bacteria
unused_rows = set(range(0, D.shape[0])).difference(used_rows)
for row in unused_rows:
    bacteria_id = bacteria_ids[row]
    bacteria[bacteria_id]['disappeared'] += 1
    if bacteria[bacteria_id]['disappeared'] > max_disappeared:
        del bacteria[bacteria_id]
```

- **Remove Lost Bacteria**: Increases the disappearance count and deletes bacteria that have disappeared for too long.

---

```python
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
```

- **Register New Bacteria**: Adds new bacteria for unmatched centroids.

---

```python
num_bacteria = len(bacteria)
```

- **Count Bacteria**: Stores the number of tracked bacteria.

---

```python
# Variables to calculate average speed of all bacteria
total_average_speed = 0
count_bacteria_with_min_lifetime = 0
max_speed = -1
fastest_bacteria_id = None

for bacteria_id, data in bacteria.items():
    lifetime = data['lifetime']
    if lifetime >= min_lifetime:
        total_time = (lifetime - 1) / fps  # Total time in seconds
        if total_time > 0:
            average_speed = (data['total_distance'] / total_time)
            data['average_speed'] = average_speed  # Store average speed
        else:
            data['average_speed'] = 0

        total_average_speed += data['average_speed']
        count_bacteria_with_min_lifetime += 1

        if data['average_speed'] > max_speed:
            max_speed = data['average_speed']
            fastest_bacteria_id = bacteria_id

if count_bacteria_with_min_lifetime > 0:
    overall_average_speed = total_average_speed / count_bacteria_with_min_lifetime
else:
    overall_average_speed = 0
```

- **Speed Calculations**: Computes the average speed for bacteria that have existed long enough and identifies the fastest bacterium.

---

```python
# Display the number of bacteria and average speed on the 'Frame' window
cv2.putText(frame, f"Bacteria Count: {num_bacteria}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(frame, f"Avg Speed: {overall_average_speed:.2f} px/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```

- **Display Information**: Draws the number of bacteria and their average speed on the video frame.

---

```python
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
        cv2.putText(frame, f"Avg Speed: {average_speed:.2f} px/s", (positions[-1][0] + 10, positions[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        cv2.circle(frame, positions[-1], 5, color, -1)
        cv2.circle(thresh, positions[-1], 5, color, -1)
```

- **Highlight Fastest Bacterium**: Draws circles and displays speed for bacteria, using a different color for the fastest one.

---

```python
# Display progress on the 'Frame' window
cv2.putText(frame, f"Progress: {progress_percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Save the processed frame to the video file
out.write(frame)

# Display the frames
cv2.imshow('Frame', frame)

# Speed up the video by reducing the wait time
key = cv2.waitKey(frame_time_ms // 2)  # Faster playback
if key == 27:  # Press 'ESC' to exit
    break
```

- **Display Progress and Save Frame**: Shows progress and writes the frame to the output video. Also displays the frame in a window and waits for a key press.

---

```python
# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_filename}")
```

- **Cleanup**: Releases video capture and writer resources and closes all OpenCV windows. Outputs the file name of the saved video.