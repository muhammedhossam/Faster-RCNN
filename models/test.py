# import torch
# import tensorflow as tf
# from torch import dtype
#
# grid_w = 50
# grid_h = 37
#
# # get the shifts in X axis [0, 1, . . . , W_feat - 1] * stride_W
# shifts_x = torch.arange(0, grid_w, dtype=torch.int32)
#
# # get the shifts in Y axis [0, 1, . . . , H_feat - 1] * stride_h
# shifts_y = torch.arange(0, grid_h, dtype=torch.int32)
#
# shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
#
# # (H_feat, W_feat)
# shifts_x = shifts_x.reshape(-1)
# shifts_y = shifts_y.reshape(-1)
# shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim = 1)
#
# print(shifts.shape)
#
#
#
#
#
#
#
#
#
# #
# # # get the shifts in X axis [0, 1, . . . , W_feat - 1] * stride_W
# # shifts_x = tf.range(0, grid_w, dtype=tf.int32)
# #
# # # get the shifts in Y axis [0, 1, . . . , H_feat - 1] * stride_h
# # shifts_y = tf.range(0, grid_h, dtype=tf.int32)
# #
# # shifts_y, shifts_x = tf.meshgrid(shifts_y, shifts_x, indexing='ij')
# #
# # # (H_feat, W_feat)
# # shifts_x = tf.reshape(shifts_x, (-1))
# # shifts_y = tf.reshape(shifts_y, (-1))
# # shifts = tf.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis = 1)
# # shifts = tf.reshape(shifts, (grid_h * grid_w,1, 4))
# #
# # print(shifts)
#
#
# # scales = torch.as_tensor([128, 256, 512], dtype=torch.float32)
# # aspect_ratio = torch.as_tensor([0.5, 1, 2])
# #
# # h_ratio =  torch.sqrt(aspect_ratio)
# # w_ratio = 1 / h_ratio
# # ws = (w_ratio[:, None] * scales[None, :]).view(-1)
# # hs = (h_ratio[:, None] * scales[None, :]).view(-1)
# #
# # base_anchor = torch.stack([-ws, -hs, ws, hs], dim = 1)
# # base_anchor = base_anchor.round()
#
#
#
# scales = tf.convert_to_tensor([128, 256, 512], dtype=tf.float32)
# aspect_ratio = tf.convert_to_tensor([0.5, 1, 2])
#
# h_ratio =  tf.sqrt(aspect_ratio)
# w_ratio = 1 / h_ratio
# ws = (w_ratio[:, None] * scales[None, :])
# hs = (h_ratio[:, None] * scales[None, :])
#
# base_anchor = tf.stack([-ws, -hs, ws, hs], axis=2) / 2
# base_anchor = tf.reshape(tf.round(base_anchor), (9, 4))
#
# shifts = tf.cast(shifts, dtype=tf.float32)
#
# anchors = tf.add(tf.reshape(shifts, (-1, 1, 4)),
#                  tf.reshape(base_anchor, (1, -1, 4))
#                  )
# print(anchors.shape)

import numpy as np
arr = np.array(
    [
        [1 , 2], [2, 44], [15, 7], [9, 3]
    ]
)
print(arr.shape)
print('=============')

print(arr[:, 0])
print('=============')

print(arr[1, ...])