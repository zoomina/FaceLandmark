import matplotlib.image as mpimg
import numpy as np
import os

data = np.load('./Face_images_with_Marked_Landmark_Points/face_images.npz')
lst = data.files
for item in lst:
    image = data[item]
    print(image[0,0,:].shape)
    for i in range(len(image[0,0,:])):
        if not os.path.exists("./Face_images_with_Marked_Landmark_Points/face_images"):
            os.mkdir("./Face_images_with_Marked_Landmark_Points/face_images")
        mpimg.imsave("./Face_images_with_Marked_Landmark_Points/face_images/face_images{:04d}.png".format(i), image[:, :, i])