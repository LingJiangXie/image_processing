import base64
import csv
import os

filename = '/home/dany/Documents/MsCelebV1-Faces-Aligned.tsv'
outputDir = '/media/dany/_data/ms_image/'

with open(filename, 'r') as tsvF:
    reader = csv.reader(tsvF, delimiter='\t')
    i = 0
    for row in reader:
        MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

        saveDir = os.path.join(outputDir, MID)
        savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        with open(savePath, 'wb') as f:
            f.write(data)

        i += 1

        if i % 1000 == 0:
            print("Extracted {} images.".format(i))