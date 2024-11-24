import cv2
import numpy as np
from multiprocessing import Pool

# Function to split the image into patches
def split_image_into_patches(image, patch_size):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, i, j))
    return patches


# Similarity metric: Sum of Squared Differences (SSD) for color images
def ssd_color(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)

# Patch matching function for color images
def match_patch(args):
    patch1, patch_list, idx1 = args
    best_match = None
    min_ssd = float('inf')
    best_match_index = None

    for idx2, (patch2, _, _) in enumerate(patch_list):
        if idx1 != idx2 and patch1.shape == patch2.shape:  # Avoid matching to itself
            ssd_value = ssd_color(patch1, patch2)
            if ssd_value < min_ssd:
                min_ssd = ssd_value
                best_match = patch2
                best_match_index = idx2

    return best_match_index

# Parallel processing function
def parallel_patch_matching(patches, num_workers=4):
    with Pool(num_workers) as pool:
        matches = pool.map(match_patch, [(patch, patches, idx) for idx, (patch, _, _) in enumerate(patches)])
    return matches

# Segment and assign consistent colors to matched patches
def segment_image_color(image, patch_size, matched_patches):
    segmented_image = np.zeros_like(image)
    
    # Generate random colors for each unique patch match
    unique_matches = list(set([x for x in matched_patches if x is not None]))
    random_colors = [np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8) for _ in range(len(unique_matches))]

    # Create a dictionary to map patch indices to colors
    color_map = {match_idx: color for match_idx, color in zip(unique_matches, random_colors)}
    
    for (patch, i, j), match_idx in zip(patches, matched_patches):
        if match_idx is not None:
            color = color_map.get(match_idx)
            segmented_image[i:i+patch_size, j:j+patch_size] = color
    
    return segmented_image

# Draw bounding boxes around segmented areas (individual objects)
def draw_bounding_boxes(segmented_image, min_contour_area=500):
    # Convert to grayscale and find contours
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on detected objects
    output_image = segmented_image.copy()
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:  # Filter out small areas (noise)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
    return output_image

if __name__ == "__main__":
    image = cv2.imread(r'C:\New folder\dogs.jpg')
    patch_size = 70
    patches = split_image_into_patches(image, patch_size)
    matched_patches = parallel_patch_matching(patches, num_workers=4)

    segmented_image = segment_image_color(image, patch_size, matched_patches)
    detected_objects_image = draw_bounding_boxes(segmented_image, min_contour_area=500)

    cv2.imwrite('detected_objects_image.jpg', detected_objects_image)
    cv2.imshow('Detected Objects', detected_objects_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
