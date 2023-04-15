"""
This file includes code from the Deep3DFaceRecon_pytorch project, which is licensed
under the MIT License. Copyright (c) 2022 Sicheng Xu.
"""

import numpy as np
from torch import jit, no_grad, tensor
from torch import float32 as torch_float32
from os import listdir as os_listdir
from os import path as os_path
import os
from util.load_mats import load_lm3d
from mtcnn import detect_faces
from util.preprocess import align_img
from PIL import Image
from renderer.render import prepare_for_render
#import sys
from root_path import pyinstaller_root

#pyinstaller_root = sys._MEIPASS
#pyinstaller_root = './'


#facial_mask_type = {'tiger': './facial_mask/tiger.npy', 'cat': './facial_mask/cat.npy'}

def read_data(im, lm3d_std, to_tensor=True, max_size = 224):
    # to RGB
    img_size = im.size
    max_img_size = max(img_size)
    if max_img_size > max_size:
        scale = max_size / max_img_size
        new_s1, new_s2 = round(img_size[0] * scale), round(img_size[1] * scale)
        small_im = im.resize((new_s1, new_s2))
    else:
        small_im = im
    small_img_size = small_im.size
    bounding_boxes, landmarks = detect_faces(small_im)

    # Ensure that the program will not exit if an incorrect input is given.
    if landmarks == []:
        landmarks = np.array([[68.7, 95, 86, 67, 91, 67, 69, 86, 97, 99]]) / 100 * min(small_img_size)

    horizontal = landmarks[:, :5].reshape((-1, 1)) * (img_size[0] / small_img_size[0])
    vertical = landmarks[:, 5:].reshape((-1, 1)) * (img_size[1] / small_img_size[1])
    combine = np.hstack((horizontal, vertical))
    landmarks = np.round(combine)

    im = im.convert('RGB')
    W,H = im.size
    lm = landmarks[:5, :]
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _, recon_info = align_img(im, lm, lm3d_std, target_size = 224) 
    if to_tensor:
        #im_for_predict = im.resize((224, 224))
        im = tensor(np.array(im)/255., dtype = torch_float32).permute(2, 0, 1).unsqueeze(0)
        #lm = torch.tensor(lm).unsqueeze(0)
    recon_info.update({'org_w': W, 'org_h': H})
    return im, recon_info 

#def compute_visuals(input_img, coeffs, vec2_3D, renderer, device='cpu'):
#    with torch.no_grad():
#
#        shape, texture = vec2_3D(coeffs['id'], coeffs['exp'], coeffs['tex'])
#        pred_mask, pred_human = renderer(shape, texture, coeffs['angle'], coeffs['trans'], coeffs['gamma'])
#        
#        output_vis_human = pred_human * pred_mask + (1 - pred_mask) * input_img[0].numpy().transpose((1, 2, 0)) * 255
#        output_vis_human = np.clip(output_vis_human, 0, 255)
#
#        return Image.fromarray(np.uint8(output_vis_human))#, Image.fromarray(np.uint8(human_1)), Image.fromarray(np.uint8(human_2)), Image.fromarray(np.uint8(animal_1)), Image.fromarray(np.uint8(animal_2))

def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:80]
    exp_coeffs = coeffs[80: 144]
    tex_coeffs = coeffs[144: 224]
    angles = coeffs[224: 227]
    gammas = coeffs[227: 254]
    translations = coeffs[254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def load_facial_mask(path):
    mask = {}
    filenames = [i for i in os_listdir(path) if i[-3:] == 'npy']
    for i in filenames:
        mask_key = i.split('.')[0]
        mask_value = np.load(os_path.join(path, i))
        mask[mask_key] = mask_value
    return mask

def load_model():
    device = 'cpu'
    # Loading model
    model_path = os_path.join(pyinstaller_root, './checkpoints/im_3d_torchscript.pth')
    model = jit.load(model_path)

    lm3d_std = load_lm3d(os_path.join(pyinstaller_root, './BFM'))
    vec2_3D, renderer = prepare_for_render(focal=1015, center=112., device=device, camera_distance=10., init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]), recenter=True, znear=5., zfar=15., bfm_file=os_path.join(pyinstaller_root, './BFM/bfm_compressed.npz'), aa_factor = 1)

    mask = load_facial_mask(os_path.join(pyinstaller_root, './facial_mask'))

    def Run(im):#, facial_mask_key):
        #assert facial_mask_key in mask.keys(), 'No such mask!'
        #facial_mask = mask[facial_mask_key]

        im = im.convert('RGB')
        im_tensor, recon_info = read_data(im, lm3d_std)

        with no_grad():
            output_coeff = model(im_tensor)[0].numpy()

        pred_coeffs = split_coeff(output_coeff)
        #output_im_human, human_1, human_2, animal_1, animal_2 = compute_visuals(im_tensor, pred_coeffs, renderer, facial_mask)
        shape, exp, texture = vec2_3D(pred_coeffs['id'], pred_coeffs['exp'], pred_coeffs['tex'])

        angle, trans, gamma = pred_coeffs['angle'], pred_coeffs['trans'], pred_coeffs['gamma']
        pred_mask, pred_human, norm = renderer(shape, exp, texture, pred_coeffs['angle'], pred_coeffs['trans'], pred_coeffs['gamma'])
        output_vis_human = pred_human * pred_mask + (1 - pred_mask) * im_tensor[0].numpy().transpose((1, 2, 0)) * 255
        pred_human = pred_human * pred_mask + (1 - pred_mask) * 255

        pred_human = np.clip(pred_human, 0, 255)
        pred_human = Image.fromarray(np.uint8(pred_human))

        output_vis_human = np.clip(output_vis_human, 0, 255)

        output_im_human = Image.fromarray(np.uint8(output_vis_human))#, Image.fromarray(np.uint8(human_1)), Image.fromarray(np.uint8(human_2)), Image.fromarray(np.uint8(animal_1)), Image.fromarray(np.uint8(animal_2))

            #output_im_human = compute_visuals(im_tensor, pred_coeffs, vec2_3D, renderer, facial_mask)
            #output_im_animal, output_im_human, v, f, human_tex, animal_tex = compute_visuals(im_tensor, pred_coeffs, renderer, facial_mask)

        img_resize, left, up, target_size, org_w, org_h, new_w, new_h = np.asarray(recon_info['img_resize']).copy(), recon_info['left'], recon_info['up'], int(recon_info['target_size']), recon_info['org_w'], recon_info['org_h'], recon_info['new_w'], recon_info['new_h']
        right, down = left+target_size, up+target_size
        #output_im_animal = np.asarray(output_im_animal)
        output_im_human = np.asarray(output_im_human)
        if left < 0:
        #    output_im_animal = output_im_animal[:, -left:]
            output_im_human = output_im_human[:, -left:]
            left = 0
        if up < 0:
        #    output_im_animal = output_im_animal[-up:, :]
            output_im_human = output_im_human[-up:, :]
            up = 0
        if right > new_w:
        #    output_im_animal = output_im_animal[:, :new_w - right]
            output_im_human = output_im_human[:, :new_w - right]
            right = new_w 
        if down > new_h:
        #    output_im_animal = output_im_animal[:new_h - down, :]
            output_im_human = output_im_human[:new_h - down, :]
            down = new_h
        img_resize_human = img_resize.copy()
        #img_resize_animal = img_resize

        #img_resize_animal[up:down, left:right] = output_im_animal
        img_resize_human[up:down, left:right] = output_im_human
        #img_resize_animal = Image.fromarray(img_resize_animal)
        img_resize_human = Image.fromarray(img_resize_human)
        #img_recon_animal = img_resize_animal.resize((org_w, org_h), resample=Image.BICUBIC)
        img_recon_human = img_resize_human.resize((org_w, org_h), resample=Image.BICUBIC)

        #return img_recon_animal, img_recon_human, v, f, human_tex, animal_tex
        return img_recon_human, pred_human, shape, exp, texture, angle, trans, gamma, norm#, human_1, human_2, animal_1, animal_2

    def shape2img(shape, exp, texture, angle, trans, gamma, norm = None):
        pred_mask, pred_human, new_norm = renderer(shape, exp, texture, angle, trans, gamma, norm, height = 448, width = 448)
        pred_human = np.clip(pred_human, 0, 255)
        pred_human = pred_mask * pred_human + (1-pred_mask) * 255
        return Image.fromarray(np.uint8(pred_human)), new_norm

    return Run, shape2img


