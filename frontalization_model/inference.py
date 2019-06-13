import argparse
import os, glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision import transforms as trans
from GAN_model_3rd import Discriminator
from GAN_model_3rd import Generator
import datetime
from PIL import Image

parser = argparse.ArgumentParser('Face Frontalization Inference')
parser.add_argument('--noise_dim', type=int, default=100)
parser.add_argument('--outf', type=str, default='./output_res')
parser.add_argument('--weight', type=str, default='netG_30000.pth')
parser.add_argument('--pose_num', type=int, default=9, help='Total training pose.')
parser.add_argument('--light_num', type=int, default=20, help='Total training lightmination.')

args = parser.parse_args()


def image_preprocess(input_image):
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
        img = img.convert('RGB').resize((96, 96), Image.ANTIALIAS)
        img = np.asarray(img)
        return img
    elif input_image.mode == 'RGB':
        img = input_image.resize((96, 96), Image.ANTIALIAS)
        img = np.asarray(img)
        return img
    else:
        raise TypeError('Confirm the input image type')


def inference(input_image, noise_dim, pose_dim, light_dim, image_name):
    """
    input_image: [H,W,C] numpy array

    """
    # preprocess the input image
    if isinstance(input_image, np.ndarray):
        img = torch.from_numpy(input_image.transpose((2, 0, 1)))
    else:
        raise TypeError('input image should be numpy array')
    # do normalization to zero mean and 1 std
    img = img.float().div(255)
    img = img.mul_(2).add_(-1)
    # create the BS dimension
    loader = trans.ToTensor()
    input_image = loader(input_image).float().unsqueeze(0)
    input_image = input_image.cuda()
    # create the random noise
    noise = torch.FloatTensor(1, noise_dim).uniform_(-1, 1).cuda()
    # create pose code for control the output code
    pose = torch.zeros(1, pose_dim).cuda()
    pose[0][3] = 1  # 3 indicate the frontal face, =1 means activate it
    # create light code for control the light condition
    light = torch.zeros(1, light_dim).cuda()
    light[0][7] = 1  # 7 indicate the standard light condition, =1 means activate it
    # generate the fake face
    face = netG(input_image, noise, pose, light)
    vutils.save_image(face,
                      '%s/frontal_face_%s.png' % (args.outf, image_name), normalize=True)


# define the network structure
netG = Generator(3, 320, 32, args.noise_dim, args.pose_num, args.light_num)
# load pretrained weight from the local file
netG.load_state_dict(torch.load(args.weight))
netG = netG.cuda().eval()


def main():
    # load images
    # detection_faces =
    detection_faces = os.path.join('/home/yiwei/Desktop/face_me_test', 'crop_image.png')
    faces = glob.glob(detection_faces)
    print(len(faces))
    for face in faces:
        image = Image.open(face)
        input_to_frontal = image_preprocess(image)
        # all inputs are numpy array, shape in (96, 96, 3)
        inference(input_to_frontal, noise_dim=args.noise_dim, pose_dim=args.pose_num, light_dim=args.light_num, image_name=os.path.basename(face))


if __name__ == "__main__":
    main()