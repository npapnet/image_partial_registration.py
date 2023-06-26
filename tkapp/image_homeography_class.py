#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessing:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.roi = None
        self.dst = None

    def select_roi(self):
        self.r = cv2.selectROI('Select ROI', self.img1)
        self.roi = self.img1[int(self.r[1]):int(self.r[1] + self.r[3]), int(self.r[0]):int(self.r[0] + self.r[2])]
        cv2.destroyWindow('Select ROI')

    def perform_homeography(self):
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.roi, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)

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
            h, w = self.roi.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            self.dst = cv2.perspectiveTransform(pts, M)
            self.img2 = cv2.polylines(self.img2, [np.int32(self.dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), 10))
            self.dst = None

    def calculate_translated_image(self):
        if self.dst is not None:
            dest_pts = np.float32([[self.r[0], self.r[1]], [self.r[0], self.r[1] + self.r[3]],
                                   [self.r[0] + self.r[2], self.r[1] + self.r[3]],
                                   [self.r[0] + self.r[2], self.r[1]]]).reshape(-1, 1, 2)

            M_inverse, _ = cv2.findHomography(np.array(self.dst), dest_pts, cv2.RANSAC, 5.0)

            new_img2 = self.img2.copy()
            h, w = self.img2.shape

            self.translated_img2 = cv2.warpPerspective(new_img2, M_inverse, (w, h))
        else:
            self.translated_img2 = None

    def plot_superimposed_images(self):
        if self.translated_img2 is not None:
            img1_color = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
            translated_img2_color = cv2.cvtColor(self.translated_img2, cv2.COLOR_GRAY2BGR)

            h1, w1 = img1_color.shape[:2]
            h2, w2 = translated_img2_color.shape[:2]

            crop_h = min(h1, h2)
            crop_w = min(w1, w2)

            cropped_img1_color = img1_color[:crop_h, :crop_w]
            cropped_translated_img2_color = translated_img2_color[:crop_h, :crop_w]

            superimposed_img = cv2.addWeighted(cropped_img1_color, 0.5, cropped_translated_img2_color, 0.5, 0)

            plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("No translated image available.")

if __name__ == "__main__":
    img1 = cv2.imread('data/im0-1kg1.jpg', 0)
    img2 = cv2.imread('data/im1-1kg1.jpg', 0)

    image_processing = ImageProcessing(img1, img2)
    image_processing.select_roi()
    image_processing.perform_homeography()
    image_processing.calculate_translated_image()
    image_processing.plot_superimposed_images()
# %%
