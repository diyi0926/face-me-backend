import cv2
from PIL import Image
import numpy as np
import torch
import time
from torchvision import transforms as trans
from detection_model.network import PNet, ONet
from detection_model.predict_m import detect
from frontalization_model.GAN_model_3rd import Generator


def detect_faces(frame):
    """
    from video frame image detect human faces and return cropped faces as well as anchor coordinates
    :param frame: numpy array(BGR)
           time:
    :return: face: 96x96x3(RGB) numpy array
             time:
             bbox_coordinate: [x1, y1, x2, y2]
             anchor_coordinate: ldmk(5,2) [[eye_x1, eye_y1],
                                          [eye_x2, eye_y2],
                                          [nose_x1, nose_y1],
                                          [mouse_x1, mouse_y1],
                                          [mouse_x2, mouse_y2]]
    """

    # load model
    pnet, onet = PNet(), ONet()
    pnet.load_state_dict(torch.load('detection_model/weight/msos_pnet_rotate.pt', map_location=lambda storage, loc: storage),strict=False)
    onet.load_state_dict(torch.load('detection_model/weight/msos_onet_rotate.pt', map_location=lambda storage, loc: storage),strict=False)
    pnet.float().cuda().eval()
    onet.float().cuda().eval()

    # detection
    # face is RGB numpy array
    face, bbox_coordinate, anchor_coordinate = detect(frame, pnet, onet, use_gpu=True)
    return face, bbox_coordinate, anchor_coordinate


def generate_frontal_face(cropped_image):
    """
    process the cropped image, generate the synthesized frontal face
    :param cropped_image: 96x96x3 RGB numpy array
    :return: fake_face: 96x96x3 RGB numpy array
             real_image: 96x96x3 RGB numpy array
             profile_feature_vector: 1x321x6x6 tensor
             frontal_feature_vector: 1x321x6x6 tensor
    """
    # load generator model
    netG = Generator(3, 320, 32, 100, 9, 20)
    netG.load_state_dict(torch.load('frontalization_model/netG_60000.pth'))
    netG = netG.cuda().eval()
    if isinstance(cropped_image, np.ndarray):
        img = torch.from_numpy(cropped_image.transpose((2, 0, 1)))
    else:
        raise TypeError('input image should be numpy array')
    # do normalization to zero mean and 1 std
    img = img.float().div(255)
    img = img.mul_(2).add_(-1)
    input_image = img.float().unsqueeze(0)
    with torch.no_grad():
        input_image = input_image.cuda()
        # create the random noise
        noise = torch.FloatTensor(1, 100).uniform_(-1, 1).cuda()
        # create pose code for control the output code
        pose = torch.zeros(1, 9).cuda()
        pose[0][3] = 1  # 3 indicate the frontal face, =1 means activate it
        # create light code for control the light condition
        light = torch.zeros(1, 20).cuda()
        light[0][7] = 1  # 7 indicate the standard light condition, =1 means activate it
        fake_face = netG(input_image, noise, pose, light)
        # fake_face and real_face
        frontal_face = fake_face[0]
        im = frontal_face.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # real face = real face
        profile_feature_vector = netG.feature(input_image)
        profile_feature_vector = profile_feature_vector.cpu().numpy()
        return im, profile_feature_vector