import os
import cv2
from scipy import misc
import copy

targetdir=os.listdir('/home/dany/Documents/datasets/lfw_112_96/')

count=0
for allDir in targetdir:
    child = os.path.join('%s%s' % ('/home/dany/Documents/datasets/lfw_112_96/', allDir))
    childx = os.listdir(child)
    for imagex in childx:
        count+=1
        img_path = os.path.join('%s%s%s' % ('/home/dany/Documents/datasets/lfw_112_96/', allDir + '/', imagex))
        img = cv2.imread(img_path)
        size = img.shape
        iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制

        h = size[0]
        w = size[1]
        for i in range(h):  # 元素循环
            for j in range(w):
                iLR[i, w - 1 - j] = img[i, j]  # 注意这里的公式没，是不是恍然大悟了（修改这里） 


        cv2.imwrite(os.path.join('%s%s%s' % ('/home/dany/Documents/datasets/lfw_112_96/', allDir + '/', 'a'+imagex)), iLR, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(count)


'''
#one image test
img = cv2.imread('/home/dany/Desktop/CASIA-WebFace_grey/0000144/020.jpg')
size = img.shape
iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
h = size[0]
w = size[1]
for i in range(h):  # 元素循环
    for j in range(w):
        iLR[i, w - 1 - j] = img[i, j]  # 注意这里的公式没，是不是恍然大悟了（修改这里）

cv2.imshow('image',iLR)
cv2.waitKey(0)

'''