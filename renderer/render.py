import numpy as np
#import  torch
#import torch.nn.functional as F
#from scipy.io import loadmat

from .Sim3DR import rasterize

def prepare_for_render(focal, center, device, camera_distance, init_lit, recenter, znear, zfar, bfm_file, aa_factor = 1):

    model = np.load(bfm_file)
    #model = loadmat(bfm_file)

    # mean face shape. [3*N,1]
    mean_shape = model['meanshape'].astype(np.float32)
    # identity basis. [3*N,80]
    id_base = model['idBase'].astype(np.float32)
    # expression basis. [3*N,64]
    exp_base = model['exBase'].astype(np.float32)
    # face indices for each vertex that lies in. starts from 0. [N,8]
    point_buf = model['point_buf'].astype(np.int64) - 1
    # vertex indices for each face. starts from 0. [F,3]
    face_buf = model['tri'].astype(np.int64) - 1
    # mean face texture. [3*N,1] (0-255)
    mean_tex = model['meantex'].astype(np.float32)[0]
    # texture basis. [3*N,80]
    tex_base = model['texBase'].astype(np.float32)#[0]

    if recenter:
        mean_shape = mean_shape.reshape([-1, 3])
        mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
        mean_shape = mean_shape.reshape([-1, 1])
    SH_a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
    SH_c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]

#    def perspective_projection(focal, center):
#        # return p.T (N, 3) @ (3, 3) 
#        return np.array([
#            focal, 0, center,
#            0, focal, center,
#            0, 0, 1
#        ]).reshape([3, 3]).astype(np.float32).transpose()
#   
#    persc_proj = perspective_projection(focal, center)
    init_lit = init_lit.reshape([1, -1]).astype(np.float32)

#    mean_shape = torch.tensor(mean_shape).to(device)
#    mean_tex = torch.tensor(mean_tex).to(device)
#    tex_base = torch.tensor(tex_base).to(device)
#    id_base = torch.tensor(id_base).to(device)
#    exp_base = torch.tensor(exp_base).to(device)
    #point_buf_torch = torch.tensor(point_buf).to(device)
    ##face_buf = torch.tensor(face_buf).to(device)
    #SH_a = torch.tensor(SH_a).to(device)
    #SH_c = torch.tensor(SH_c).to(device)
    #persc_proj = torch.tensor(persc_proj).to(device)
    #init_lit = torch.tensor(init_lit).to(device)

    #fov = 2 * np.arctan(center / focal) * 180 / np.pi

#    def compute_shape_test(id_coeff, exp_coeff):
#        """
#        Return:
#            face_shape       -- torch.tensor, size (B, N, 3)
#
#        Parameters:
#            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
#            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
#        """
#        batch_size = id_coeff.shape[0]
#        id_part = torch.einsum('ij,aj->ai', id_base, id_coeff)
#        exp_part = torch.einsum('ij,aj->ai', exp_base, exp_coeff)
#        face_shape = id_part + mean_shape.reshape([1, -1])
#        return face_shape.reshape([batch_size, -1, 3]), exp_part.reshape([batch_size, -1, 3])

    def compute_shape(id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        #print(id_base.shape, id_coeff.shape)
        id_part = id_base.dot(id_coeff)
        exp_part = exp_base.dot(exp_coeff)
        face_shape = id_part + mean_shape.reshape(-1)
        return face_shape.reshape([-1, 3]), exp_part.reshape([-1, 3])
   
    def compute_norm(face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[face_buf[:, 0]]
        v2 = face_shape[face_buf[:, 1]]
        v3 = face_shape[face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = np.cross(e1, e2)
        #norm = np.linalg.norm(face_norm, axis = 1)
        #face_norm = face_norm / norm.reshape((-1, 1))
        face_norm = np.vstack((face_norm, np.zeros((1, 3))))
        
        #print(face_norm.shape, point_buf.shape)
        #print(face_norm[point_buf].shape)
        #print(face_norm[point_buf_torch].shape)
        #print(np.sum((face_norm[point_buf] - face_norm[point_buf_torch])**2))

        vertex_norm = np.sum(face_norm[point_buf], axis=1)
        #vertex_norm_test = np.sum(face_norm[torch.tensor(point_buf)], axis=1)
        #print(np.sum((vertex_norm - vertex_norm_test)**2))
        norm = np.linalg.norm(vertex_norm, axis = 1)
        vertex_norm = vertex_norm / norm.reshape((-1, 1))
        return vertex_norm

    def compute_color(face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        #gamma = gamma.numpy()[0]
        v_num = face_texture.shape[0]
        #face_norm = face_norm.numpy()[0]
        face_norm_c1, face_norm_c2, face_norm_c3 = face_norm[:, :1], face_norm[:, 1:2], face_norm[:, 2:]

        a, c = SH_a, SH_c
        gamma = gamma.reshape([3, 9])
        gamma = gamma + init_lit
        gamma = gamma.T

        Y = np.hstack([
             a[0] * c[0] * np.ones_like(face_norm_c1),
            -a[1] * c[1] * face_norm_c2,
             a[1] * c[1] * face_norm_c3,
            -a[1] * c[1] * face_norm_c1,
             a[2] * c[2] * face_norm_c1 * face_norm_c2,
            -a[2] * c[2] * face_norm_c2 * face_norm_c3,
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm_c3 ** 2 - 1),
            -a[2] * c[2] * face_norm_c1 * face_norm_c3,
            0.5 * a[2] * c[2] * (face_norm_c1 ** 2  - face_norm_c2 ** 2)
        ])
        r = Y.dot(gamma[:, :1])
        g = Y.dot(gamma[:, 1:2])
        b = Y.dot(gamma[:, 2:])
        face_color = np.hstack([r, g, b]) * face_texture
#        face_color = torch.tensor(face_color.copy())
        return face_color

    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        x, y, z = angles[0], angles[1], angles[2]
        
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)], 
            [0, np.sin(x), np.cos(x)]
        ])#.reshape([batch_size, 3, 3])
        
        rot_y = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])#.reshape([batch_size, 3, 3])

        rot_z = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])#.reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.T#.permute(0, 2, 1)


    def to_camera(face_shape):
#        print(face_shape.shape)
#        face_shape = face_shape.numpy()[0]
#
        face_shape[..., -1] = camera_distance - face_shape[..., -1]
        return face_shape

#    def to_image(face_shape):
#        """
#        Return:
#            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction
#
#        Parameters:
#            face_shape       -- torch.tensor, size (B, N, 3)
#        """
#        # to image_plane
#        face_proj = face_shape @ persc_proj
#        face_proj = face_proj[..., :2] / face_proj[..., 2:]
#
#        return face_proj


    def transform(face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape.dot(rot) + trans#.unsqueeze(1)

    def compute_perspective_proj(v):
        assert len(v.shape) == 2
        assert type(v) == np.ndarray
        v[:, :2] = focal/center * v[:, :2] / v[:,2].reshape((-1, 1))
        v = (v + np.array((1, 1, 0)))
        v[:, 2] = -v[:, 2]
        return v

    def compute_texture(tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        face_texture = tex_base.dot(tex_coeff) + mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([-1, 3])#, face_texture[0]

    def vec2_3D(coef_id, coef_exp, coef_tex):#, facial_mask):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        face_shape, face_exp = compute_shape(coef_id, coef_exp)
        face_texture = compute_texture(coef_tex)
        #return torch.tensor(face_shape).unsqueeze(0), torch.tensor(face_exp).unsqueeze(0), torch.tensor(face_texture).unsqueeze(0)
        return face_shape, face_exp, face_texture

    def render(face_shape, face_exp, face_texture, coef_angle, coef_trans, coef_gamma, face_norm = None, height = 224, width = 224):
        #face_shape = face_shape.numpy()[0]
        #face_exp = face_exp.numpy()[0]
        #face_texture = face_texture.numpy()[0]

        face_shape = face_shape + face_exp
        
        if face_norm is None:
            face_norm = compute_norm(face_shape)

        rotation = compute_rotation(coef_angle)

        face_shape_transformed = transform(face_shape, rotation, coef_trans)
        face_vertex = to_camera(face_shape_transformed)


        face_norm_roted = face_norm.dot(rotation)

        face_color = compute_color(face_texture, face_norm_roted, coef_gamma)
        f_color = np.clip(face_color, 0, 1)

        #mask_texture = torch.from_numpy(facial_mask[np.newaxis, ...]).to(device=face_vertex.device) / 255.
        #mask_color = compute_color(mask_texture, face_norm_roted, coef_gamma)
        #m_color = np.clip(mask_color[0].numpy(), 0, 1)
        
        v = compute_perspective_proj(face_vertex)

        #factor = aa_factor
        factor = 1
        #h, w = 224, 224
        h, w = height, width
        assert h==w
        h *= factor
        w *= factor

        v[:, :2] = v[:, :2] / 2 * h - 0.5

        #pred_animal, pred_mask, pred_depth = rasterize(v.astype(np.float32), face_buf.astype(np.int32).copy(order='C'), m_color, height=224, width=224)
        #pred_animal = pred_animal.transpose((2, 0, 1))[np.newaxis, ...] / 255.

        pred_human, pred_mask, pred_depth = rasterize(v.astype(np.float32), face_buf.astype(np.int32).copy(order='C'), f_color, height=h, width=w)
        #pred_human = torch.from_numpy(pred_human).permute((2, 0, 1)).unsqueeze(0)
        #pred_mask = torch.from_numpy(pred_mask[np.newaxis, np.newaxis, ...].copy())
        #pred_mask = torch.nn.functional.avg_pool2d(pred_mask, kernel_size=factor, stride=factor).numpy()[0, 0]
        #pred_mask = (pred_mask == 1)
        #pred_mask = pred_mask[..., np.newaxis]
        #pred_human = torch.nn.functional.avg_pool2d(pred_human.to(dtype=torch.float32), kernel_size=factor, stride=factor)[0].permute((1, 2, 0)).numpy()# / 255.

        return pred_mask[..., np.newaxis], pred_human, face_norm

    return vec2_3D, render



