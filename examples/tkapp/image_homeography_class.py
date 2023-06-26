#%%
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
class ImageProcessing:
    MIN_GOOD_MATCHES = 10 
    AUTOCLOSE_ROI_SELECTION_MS = 10000 # in milli seconds

    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.roi = None
        self.dst = None

    def select_roi(self):
        self.r = cv2.selectROI('Select ROI', self.img1)
        self.roi = self.img1[int(self.r[1]):int(self.r[1] + self.r[3]), int(self.r[0]):int(self.r[0] + self.r[2])]
        cv2.waitKey(self.AUTOCLOSE_ROI_SELECTION_MS)
        cv2.destroyWindow('Select ROI')

    def perform_homeography(self):
        # this step follow roi selection
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.roi, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # find good matches
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > self.MIN_GOOD_MATCHES:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = self.roi.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            self.dst = cv2.perspectiveTransform(pts, M)
            self.img2 = self._add_polylines()
        else:
            print("Not enough matches are found - {}/{}".format(len(good), 10))
            self.dst = None

    def _add_polylines(self):
        """adds polylines to the second image
        """        
        return cv2.polylines(self.img2, [np.int32(self.dst)], True, 255, 3, cv2.LINE_AA)

    def calculate_translated_image(self, onlyTranslationFlag:bool=True):
        if self.dst is not None:
            dest_pts = np.float32([[self.r[0], self.r[1]], [self.r[0], self.r[1] + self.r[3]],
                                   [self.r[0] + self.r[2], self.r[1] + self.r[3]],
                                   [self.r[0] + self.r[2], self.r[1]]]).reshape(-1, 1, 2)

            M_inverse = self.calc_homeography(dest_pts, onlyTranslationFlag=onlyTranslationFlag)

            new_img2 = self.img2.copy()
            h, w = self.img2.shape

            self.translated_img2 = cv2.warpPerspective(new_img2, M_inverse, (w, h))
        else:
            self.translated_img2 = None

    def calc_homeography(self, dest_pts, onlyTranslationFlag:bool=True):
        """calculates homeography matrix

        Args:
            dest_pts (_type_): _description_

        Returns:
            _type_: _description_
        """        
        M_inverseraw, _ = cv2.findHomography(np.array(self.dst), dest_pts, cv2.RANSAC, 5.0)
        M_inverse = M_inverseraw.copy()
        # mimimize shear 
        if onlyTranslationFlag:
            M_inverse = np.eye(3)
            M_inverse[0,2] = M_inverseraw[0,2]
            M_inverse[1,2] = M_inverseraw[1,2]
        return M_inverse

    def plot_superimposed_images(self):
        if self.translated_img2 is not None:
            img1_color = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
            translated_img2_color = cv2.cvtColor(self.translated_img2, cv2.COLOR_GRAY2BGR)

            cropped_img1, cropped_img2 = self._crop_images(img1_color, translated_img2_color)

            superimposed_img = cv2.addWeighted(
                cropped_img1, 0.5, 
                cropped_img2, 0.5, 0)

            plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("No translated image available.")

    def _crop_images(self, img_1, img_2):
        """auxilliary function that crops two images and  them to the smallest size
    
        Args:
            img_1 (_type_): image 1
            img_2 (_type_): image 2

        Returns:
            _type_: _description_
        """        
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]

        crop_h = min(h1, h2)
        crop_w = min(w1, w2)

        cropped_img1 = img_1[:crop_h, :crop_w]
        cropped_img2 = img_2[:crop_h, :crop_w]
        return cropped_img1,cropped_img2

if __name__ == "__main__":
    DATADIR = pathlib.Path('../../data/')
    img1 = cv2.imread(str(DATADIR/'im0-1kg1.jpg'), 0)
    img2 = cv2.imread(str(DATADIR/'im1-1kg1.jpg'), 0)

    image_processing = ImageProcessing(img1, img2)
    image_processing.select_roi()
    image_processing.perform_homeography()
    image_processing.calculate_translated_image(onlyTranslationFlag=False)
    image_processing.plot_superimposed_images()
# %%
