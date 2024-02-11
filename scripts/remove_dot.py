import os
import os.path as osp

folder = '/data/home/hao/endovis2017'

def remove_dot(folder):
    for f in os.listdir(folder):
        path = osp.join(folder, f)
        if f[0] == '.':# and f != "._.DS_Store":
            print(path)
            os.remove(path)
        if osp.isdir(path):
            remove_dot(path)

remove_dot(folder)
