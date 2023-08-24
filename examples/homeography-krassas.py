# this is the code for performing the homeography. 
# it uses a inline variables for the file names.
# 
# it can be used as a test bed for different proor of concepts
#
# most of the code has been transfered to image_homeography_class.py 

#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tkinter as tk
from PIL import ImageTk

# Container class to hold the original images

# minimum good matches
MIN_GOOD_MATCHES = 10 

# %%
img1 = cv2.imread('data/im0-1kg1.jpg',0)
img2 = cv2.imread('data/im1-1kg1.jpg',0)

# img1 = app.container.get_initial_image()
# img2 = app.container.get_final_image()

#%%
# Select ROI in the first image
r = cv2.selectROI('Select ROI',img1)
roi = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.destroyWindow('Select ROI')

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(roi, None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# Homography
if len(good) > MIN_GOOD_MATCHES:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = roi.shape
    pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # plt.imshow(img2, cmap='gray')
else:
    print("Not enough matches are found - {}/{}".format(len(good),10))
    matchesMask = None
# %%
# Points in destination image that you want to map to
dest_pts = np.float32([[r[0], r[1]], 
                       [r[0], r[1]+r[3]], 
                       [r[0]+r[2], r[1]+r[3]], 
                       [r[0]+r[2], r[1]]]).reshape(-1,1,2)

# Calculate the inverse homography matrix
M_inverseraw, _ = cv2.findHomography(np.array(dst), dest_pts, cv2.RANSAC,5.0)
M_inverse = M_inverseraw
# # mimimize shear 
# M_inverse = np.eye(3)
# M_inverse[0,2] = M_inverseraw[0,2]
# M_inverse[1,2] = M_inverseraw[1,2]
print(M_inverse)
# Create a new image which is a copy of img2
new_img2 = img2.copy()

# Get the shape of img2
h, w = img2.shape

# Apply the inverse homography matrix to img2
translated_img2 = cv2.warpPerspective(new_img2, M_inverse, (w, h))

# Now translated_img2 is the translated image

# %%
def crop_images_to_same_size(img1, img2):
    # Get the shape of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the size of the area to crop to (smallest width and height)
    crop_h = min(h1, h2)
    crop_w = min(w1, w2)

    # Crop the images
    cropped_img1 = img1[:crop_h, :crop_w]
    cropped_img2 = img2[:crop_h, :crop_w]

    return cropped_img1, cropped_img2
# Convert the grayscale images to BGR
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
translated_img2_color = cv2.cvtColor(translated_img2, cv2.COLOR_GRAY2BGR)
# Crop the images to the same size
cropped_img1_color, cropped_translated_img2_color = crop_images_to_same_size(img1_color, translated_img2_color)

# Superimpose the cropped images
superimposed_img = cv2.addWeighted(cropped_img1_color, 0.5, cropped_translated_img2_color, 0.5, 0)

# Display the superimposed image
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for displaying
# plt.show()

#%% [markdown]
# # Difference
#  the following code applied differences
#%%
# # %% 
def simple_diff (img1, img2):
    # Compute the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Apply a binary threshold to the difference (this will make the "hotspots" appear white)
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return threshold

# threshold = simple_diff (img1=cropped_img1_color,img2=cropped_translated_img2_color)

def edge_detection_diff (img1,img2, threshold1=30, threshold2=100):
    edges1 = cv2.Canny(img1, threshold1=threshold1, threshold2=threshold2)
    edges2 = cv2.Canny(img2, threshold1=threshold1, threshold2=threshold2)

    # Compute the absolute difference between the edge images
    diff = cv2.absdiff(edges1, edges2)
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return threshold

threshold = edge_detection_diff (img1=cropped_img1_color,img2=cropped_translated_img2_color)




class ImageWindow(tk.Toplevel):
    def __init__(self, image, title='Image', *args, **kwargs):
        super(ImageWindow, self).__init__(*args, **kwargs)
        
        self.title(title)

        # Convert the grayscale OpenCV image to PIL format, then to ImageTk format
        img_pil = PIL.Image.fromarray(image)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)

        # Create a label to hold the image
        self.label = tk.Label(self, image=self.img_tk)
        self.label.pack()

# Display the hotspot image
# Create a new tkinter window and start the GUI
root = tk.Tk()
root.withdraw()  # hide main root window
app = ImageWindow(threshold, 'Hotspots')
# root.mainloop()
 # %%
app.mainloop()
# # %%

# %%
