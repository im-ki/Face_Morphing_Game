#import copy
from facial_mask_game import load_model
Run, shape2img = load_model()

import numpy as np
from cv2 import cvtColor, COLOR_RGB2BGR

class Face:
    def __init__(self, shape = None, exp = None, tex = None, angle = None, trans = None, gamma = None, rendered_img = None, norm = None, shape_v = None, exp_v = None, tex_v = None):
        self.shape_v      = shape_v
        self.exp_v        = exp_v
        self.tex_v        = tex_v

        self.shape        = shape
        self.exp          = exp
        self.tex          = tex

        self.org_exp = exp
        self.human_tex = tex

        self.current_face_shape = None
        self.current_face_exp = None
        self.current_face_tex = None
        self.current_face_gamma = None

        self.angle        = angle
        self.trans        = trans
        self.gamma        = gamma

        self.rendered_img = rendered_img
        self.norm         = norm

    def horizontal_rot(self, angle):
        self.angle[1] = angle
        self.update_rendered_img()
        return self.rendered_img

    def vertical_rot(self, angle):
        self.angle[0] = -angle
        self.update_rendered_img()
        return self.rendered_img

    #def rot(self, h_angle, v_angle):
    #    self.angle[1] = h_angle
    #    self.angle[0] = -v_angle

    def interp_shape(self, obj2, ratio):
        self.current_face_shape = ratio * obj2.shape + (1-ratio) * self.shape
        self.update_rendered_img(update_norm = True)
        return self.rendered_img

    def interp_exp(self, obj2, ratio):
        self.current_face_exp = ratio * obj2.exp + (1-ratio) * self.exp
        self.update_rendered_img(update_norm = True)
        return self.rendered_img

    def interp_tex(self, obj2, ratio):
        self.current_face_tex = ratio * obj2.tex + (1-ratio) * self.tex
        self.current_face_gamma = ratio * obj2.gamma + (1-ratio) * self.gamma
        self.update_rendered_img()
        return self.rendered_img

    def update_multiple_params(self, h_angle = None, v_angle = None, shape_ratio = None, exp_ratio = None, tex_ratio = None, obj2 = None):
        if h_angle is not None:
            self.angle[1] = h_angle
        if v_angle is not None:
            self.angle[0] = -v_angle
        if obj2 is not None:
            if shape_ratio is not None:
                self.current_face_shape = shape_ratio * obj2.shape + (1-shape_ratio) * self.shape
            if exp_ratio is not None:
                self.current_face_exp = exp_ratio * obj2.exp + (1-exp_ratio) * self.exp
            if tex_ratio is not None:
                self.current_face_tex = tex_ratio * obj2.tex + (1-tex_ratio) * self.tex
                self.current_face_gamma = tex_ratio * obj2.gamma + (1-tex_ratio) * self.gamma
        self.update_rendered_img(update_norm = True)

    def update_rendered_img(self, update_norm = False):
        if update_norm == False:
            new_left, _ = shape2img(self.current_face_shape, self.current_face_exp, self.current_face_tex, self.angle, self.trans, self.current_face_gamma, self.norm)
        else:
            new_left, self.norm = shape2img(self.current_face_shape, self.current_face_exp, self.current_face_tex, self.angle, self.trans, self.current_face_gamma)
        left_image = np.array(new_left)
        self.rendered_img = cvtColor(left_image, COLOR_RGB2BGR)

    def reset_current_state(self):
        self.current_face_shape = self.shape
        self.current_face_exp = self.exp
        self.current_face_tex = self.tex
        self.current_face_gamma = self.gamma

    def img2face(self, img):
        result, _, shape, exp, texture, angle, trans, gamma, self.norm = Run(img)
        self.shape = shape
        self.tex = texture
        self.human_tex = texture
        self.exp = exp
        self.org_exp = exp
        self.gamma = gamma
        self.angle = np.zeros(3, dtype=np.float32)

        self.current_face_shape = shape
        self.current_face_tex = texture
        self.current_face_exp = exp
        self.current_face_gamma = gamma

        #self.angle = angle
        self.trans = trans

        pred_human, _ = shape2img(self.current_face_shape, self.current_face_exp, self.current_face_tex, self.angle, self.trans, self.current_face_gamma, self.norm)
        pred_human = np.array(pred_human)
        pred_human = cvtColor(pred_human, COLOR_RGB2BGR)
        self.rendered_img = pred_human
        result = np.array(result)
        result = cvtColor(result, COLOR_RGB2BGR)
        return result


