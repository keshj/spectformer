# save as make_cars_pkl.py and run:  python make_cars_pkl.py
import scipy.io as sio, pickle, os, numpy as np

root = "../datasets"  # <- your folder
mat  = sio.loadmat(os.path.join(root, "car_devkit/devkit", "cars_train_annos.mat"))
annos = mat["annotations"][0]

rel_paths, classes, tests = [], [], []
for a in annos:
    p = a[0][0]                         # e.g. 'car_ims/000001.jpg'
    is_test = int(a[6][0][0])           # 0=train, 1=test
    # map to your actual folders (cars_train / cars_test)
    p = p.replace("car_ims/", "cars_test/" if is_test==1 else "cars_train/")
    rel_paths.append(p)
    classes.append(int(a[5][0][0]))     # 1..196
    tests.append(is_test)

def wrap_vec(lst):
    arr = np.empty((1, len(lst)), dtype=object)
    for i,v in enumerate(lst): arr[0,i] = v
    return arr

list_mat = {"annotations":{
    "relative_im_path": wrap_vec(rel_paths),
    "class":            wrap_vec(classes),
    "test":             wrap_vec(tests),
}}

with open(os.path.join(root, "cars_anno.pkl"), "wb") as f:
    pickle.dump(list_mat, f)
print("Wrote", os.path.join(root, "cars_anno.pkl"), "with", len(rel_paths), "entries")
