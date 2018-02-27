import os
import cv2
from scipy import misc


def grey(image_dir,grey_image_dir):

    if not os.path.exists(grey_image_dir) :

        os.makedirs(grey_image_dir)

        for fold in os.listdir(image_dir):

            other_sub_fold = os.path.join(grey_image_dir, fold)

            os.makedirs(other_sub_fold)

    count = 0

    for sub_fold in os.listdir(image_dir):

        sub_fold_dir = os.path.join(image_dir,sub_fold)

        other_sub_fold_dir=os.path.join(grey_image_dir,sub_fold)

        image_list = os.listdir(sub_fold_dir)

        for image in image_list:

            count += 1
            img_path = os.path.join(sub_fold_dir,image)

            img = cv2.imread(img_path)
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            misc.imsave(os.path.join(other_sub_fold_dir,image),grayImage)

            print(count)


grey('/home/deep-visage/Documents/lfw_112','/home/deep-visage/Documents/lfw_grey_112')
