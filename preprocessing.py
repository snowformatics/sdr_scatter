import os
import shutil

images = os.listdir("C:/Users/stefanie/Dropbox/sdr/captures/")
sounds = os.listdir("C:/Users/stefanie/Dropbox/sdr/sounds/")
path = "E:/sdr/meteors/positives/"


def copy_positives():

    for i in images:
        date = i.split('_')[0]
        id = i.split('_')[-1].split('.')[0]
        #print (date, id)
        for s in sounds:
            if s.startswith(date) and s.endswith(str('_' + id) + '.wav'):
                print (i, s)
                shutil.copy("C:/Users/stefanie/Dropbox/sdr/captures/" +  i, path + i)
                shutil.copy("C:/Users/stefanie/Dropbox/sdr/sounds/" + s, path + s)


copy_positives()
