
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from image_container import ImageContainer
import numpy as np
import matplotlib.pyplot as plt
# Container class to hold the original images

from image_display_classes import tkFrameOriginalImages
#%%


class ImageManipulationWindow(tk.Toplevel):
    def __init__(self, container):
        super().__init__()
        self.title("Image Manipulation")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.container = container
        self.visible = False

        self.select_area_button = tk.Button(self, text="Select Area", command=self.select_area)
        self.select_area_button.grid(row=0, column=0, pady=10)

        self.apply_homeography_button = tk.Button(self, text="Apply Homeography", command=self.apply_homeography)
        self.apply_homeography_button.grid(row=0, column=1, pady=10)

        self.save_final_image_button = tk.Button(self, text="Save Final Image", command=self.save_final_image)
        self.save_final_image_button.grid(row=0, column=2, pady=10)

    def toggle_visibility(self):
        self.visible = not self.visible
        if self.visible:
            self.update_images()
            self.deiconify()
        else:
            self.withdraw()

    def update_images(self):
        initial_img = Image.fromarray(self.container.get_initial_image())
        initial_tkimg = ImageTk.PhotoImage(initial_img)
        self.initial_image_label.configure(image=initial_tkimg)
        self.initial_image_label.image = initial_tkimg

        final_img = Image.fromarray(self.container.get_final_image())
        final_tkimg = ImageTk.PhotoImage(final_img)
        self.final_image_label.configure(image=final_tkimg)
        self.final_image_label.image = final_tkimg

    def select_area(self):
        img1 = self.container.get_initial_image()
        r = cv2.selectROI('Select ROI', img1)
        roi = img1[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.destroyWindow('Select ROI')

        self.roi = roi

    def apply_homeography(self):
        img1 = self.container.get_initial_image()
        img2 = self.container.get_final_image()

        roi = self.roi

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(roi, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = roi.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), 10))
            matchesMask = None

        dest_pts = np.float32([[r[0], r[1]], [r[0] + r[2], r[1] + r[3]]]).reshape(-1, 1, 2)

        M_inverse, _ = cv2.findHomography(np.array(dst), dest_pts, cv2.RANSAC, 5.0)

        new_img2 = img2.copy()
        h, w = img2.shape

        translated_img2 = cv2.warpPerspective(new_img2, M_inverse, (w, h))

        self.translated_img2 = translated_img2

    def save_final_image(self):
        if hasattr(self, 'translated_img2'):
            file_path = filedialog.asksaveasfilename(title="Save Final Image",
                                                     filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
            if file_path:
                cv2.imwrite(file_path, self.translated_img2)
                print("Final image saved successfully.")
        else:
            print("No final image available.")

    def on_closing(self):
        self.toggle_visibility()

# Tkinter app class
class ImageSelectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize the image container
        self.container = ImageContainer()

        # Create a frame for original images
        self.frame_original_images = tkFrameOriginalImages(self, self.container)
        self.frame_original_images.pack()

        self.open_manipulation_button = tk.Button(self, text="Open Manipulation Window", command=self.open_manipulation_window)
        self.open_manipulation_button.pack(pady=10)

        self.manipulation_window = ImageManipulationWindow(self.container)


    def open_manipulation_window(self):
        self.manipulation_window.toggle_visibility()

if __name__ == "__main__":
    # Create an instance of the Tkinter app
    app = ImageSelectionApp()

    # Start the Tkinter main loop
    app.mainloop()
