import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

#img_dir="E:/sdr/meteors/26012023/"
log = open("C:/Users/stefanie/Dropbox/sdr/logs/event_log_202301.csv", 'r').readlines()
img_dir = "E:/sdr/meteors/test/"

cnn_model = load_model('../train_meteor_classifier/02.h5')
all_files=os.listdir(img_dir)
data_path = [os.path.join(img_dir + "/" + i) for i in all_files]


def show_images(predict):
    """Display images in matplotlib subplots."""
    k=0
    for i in data_path:

        name = i.split('/')[-1].rsplit('_', maxsplit=1)[0]
        name = name.replace('event', '')

        id = i.split('/')[-1].rsplit('_', maxsplit=1)[-1].split('.')[0]
        #print (name, id)
        k = k + 1
        # for 20 images
        plt.subplot(4,5,k)
        #plt.subplot(5, 9, k)

        plt.imshow(imread(i))
        plt.xticks([])
        plt.yticks([])
        if predict:
            roi_org = cv2.imread(i)
            roi_resized = cv2.resize(roi_org, (130, 100))
            # cv2.imshow('', roi_resized)
            # cv2.waitKey(0)
            img_tensor = image.img_to_array(roi_resized)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            # Remember that the model was trained on inputs
            # that were preprocessed in the following way:
            img_tensor /= 255.
            preds = cnn_model.predict(img_tensor)
            label = 'area:' + str(int(round(preds[0][0] * 100, 0))) + ' trail:' + str(int(round(preds[0][1] * 100, 0))) + ' head:' + str(int(round(preds[0][2] * 100, 0))) + ' point:' + str(int(round(preds[0][3] * 100, 0)))

            print(i, round(preds[0][0] * 100, 2), round(preds[0][1] * 100, 2), round(preds[0][2] * 100, 2),
                 round(preds[0][3] * 100, 2))
        else:
            for x in log:
                x = x.strip()
                x = x.split(',')
                date = x[0].replace('/', '')
                id2 = x[2]
                #print (name.split('_')[0], date)
                if name.split('_')[0] == date and id == id2:
                    label = 'dB:' + x[3] + ' Hz:' + x[6] + ' s:' + x[7]
            #print (name, label)

        plt.title(name + '_' + id, fontdict={'fontsize':10})
        plt.xlabel(label, size=10)
        print (name + id, label)
       

    plt.axis('off')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=1,
                        hspace=1)
    plt.show()


show_images(predict=True)