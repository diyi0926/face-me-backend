import os,glob
import numpy as np
from PIL import Image
from predict_m import detect
from network import PNet,ONet
import torch


pnet, onet = PNet(), ONet()
pnet.load_state_dict(torch.load('weight/msos_pnet_rotate.pt',map_location=lambda storage, loc:storage), strict=False)
onet.load_state_dict(torch.load('weight/msos_onet_rotate.pt',map_location=lambda storage, loc:storage), strict=False)
pnet.float()
onet.float()
pnet.eval()
onet.eval()



image_root = '/home/yiwei/github/e2e-joint-face-detection-and-alignment/picture'
images_set = os.path.join(image_root, 'ryan_45.jpg')
images = glob.glob(images_set)

for image in images:
    img = Image.open(image)
    faces = detect(image, pnet, onet)
    print('there are {} faces in this image'.format(len(faces)))
    for face in faces:
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        crop_img = img.crop((x1,y1,x2,y2))
        crop_img = crop_img.resize((100,100), Image.ANTIALIAS)

        # PIL Image like image
        # crop_img.show()
        crop_img.save('/home/yiwei/Desktop/face_me_test/ryan_45.png')

