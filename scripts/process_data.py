import csv
import numpy as np
from PIL import Image
import csv
import cv2
import os
import shutil

def background_subtraction(background_path, image_path, image_numbers, gt_save_path="./record_images/", margin=80, middle=260):
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)
    images = []
    for i in range(image_numbers):
        if os.path.exists(os.path.join(image_path, str(i)+".png")):
            images.append(np.array(cv2.imread(os.path.join(image_path, str(i)+".png"))))
        else:
            break

    background = np.array(Image.open(background_path))
    
    background = cv2.resize(background, (640, 360))
    background = cv2.GaussianBlur(background,(21,21),0)
    background = cv2.bilateralFilter(background,9,75,75)

    fgbg = cv2.createBackgroundSubtractorMOG2(history = 11000, varThreshold = 512, detectShadows = False)
    fgbg2 = cv2.createBackgroundSubtractorKNN(history = 11000, dist2Threshold = 1200, detectShadows = False)

    for i in range(10000):
        if i % 5000 == 4999:
            print(i)
        fgmask = fgbg.apply(background)
        fgmask2 = fgbg2.apply(background)

    cv2.imshow('frame',background)
    k = cv2.waitKey(-1) & 0xff
    for i in range(len(images)):
    #for i in range(20, 50):
        image = images[i]
        image = cv2.resize(image, (640, 360))
        image = cv2.GaussianBlur(image,(21,21),0)
        image = cv2.bilateralFilter(image,9,75,75)
        fgmask = fgbg.apply(image)
        fgmask2 = fgbg2.apply(image)
        fgmask[fgmask2 != 255] = 0
        fgmask = gt_denoise(fgmask, 2000)

        #fgmask = gt_denoise()
        im_toshow = np.zeros_like(fgmask)
        im_toshow[:, margin:-margin] = fgmask[:, margin:-margin]
        #im_toshow[:, middle:-middle] = 0
        #im_toshow[:160, :middle] = 0
        #im_toshow = fgmask

        #im_toshow, num_cc = get_prompt(im_toshow)

        # if num_cc != 3:
        #     print("image: ", i, "has cc number:", num_cc)
        cv2.imshow('frame',im_toshow)
        cv2.imwrite(os.path.join(gt_save_path,str(i)+".png"), im_toshow)
        k = cv2.waitKey(10) & 0xff

def get_prompt(img):
    #doing erode to get two connected components as prompt each for one tool.
    kernel = np.ones((2, 2), np.uint8)
    #img = cv2.erode(img, kernel, iterations=1)
    img = gt_denoise(img, 100)
    img = select_largest_cc(img, 2)
    num_cc = len(np.unique(cv2.connectedComponents(img, connectivity=4)[1]))
    while num_cc > 1 and num_cc < 3:
        img = cv2.erode(img, kernel, iterations=1)
        img = gt_denoise(img, 100)
        img = select_largest_cc(img, 2)
        num_cc = len(np.unique(cv2.connectedComponents(img, connectivity=4)[1]))
    return img, num_cc

def select_largest_cc(img, top_k = 2):
    #select the largest top_k connnected components
    _, CCs = cv2.connectedComponents(img, connectivity=4)
    sizes = {}
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        numbers = (CCs == i).sum()
        sizes[i] = numbers
    
    sizes = sorted(sizes.items(), key=lambda x:x[1])
    for i in range(len(sizes) - top_k):
        img[CCs == sizes[i][0]] = 0
    return img

def gt_denoise(image, threshold = 2000, color_flip = True):
    #remove small connnected components

    _, CCs = cv2.connectedComponents(image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < threshold:
            image[CCs == i] = 0

    if color_flip:
        _, CCs = cv2.connectedComponents(255 - image, connectivity=4)
        labels = np.unique(CCs)
        for i in labels:
            if i == 0:
                continue
            if (CCs == i).sum() < threshold:
                image[CCs == i] = 255
    return image

def get_csv(path):
    kinematics = np.load(path)
    print(kinematics.shape)
    print(kinematics[0])
    psm_1_kinematics = kinematics[:, :25]
    psm_2_kinematics = kinematics[:, 25:]
    writeKinematics("./ambf/psm_1/psm_1_kinematics_aligned.csv", psm_1_kinematics)
    writeKinematics("./ambf/psm_2/psm_2_kinematics_aligned.csv", psm_2_kinematics)

def writeKinematics(path, kinematics):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k in kinematics:
            k = list(k)
            writer.writerow(k)

def overlay_gt_to_image(combined_gt, image):
    overlayed_image = image24
    overlayed_image[:, :, 0][combined_gt > 0] = 200
    return overlayed_image

def combine_gts(gt1_path, gt2_path, image_path, gt_save_path, image_save_path, image_numbers, pre_rejects = None):
    accepts = []
    rejects = []
    gts = []
    images = []
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    i = 0
    while i < image_numbers:
        image = np.array(Image.open(image_path + str(i) + '.png'))
        gt1 = np.load(gt1_path + str(i) + '.npy')
        gt2 = np.load(gt2_path + str(i) + '.npy')
        combined_gt = (((gt1 + gt2) >= 1) * 255).astype(np.uint8)
        combined_gt = gt_denoise(combined_gt, threshold=20000)
        gts.append(combined_gt > 0)
        images.append(image)
        overlayed_image = overlay_gt_to_image(combined_gt, image)
        im_toshow = np.zeros((overlayed_image.shape[0]*2, overlayed_image.shape[1], overlayed_image.shape[2])).astype(np.uint8)
        im_toshow[:overlayed_image.shape[0], :] = overlayed_image
        im_toshow[overlayed_image.shape[0]:, :][combined_gt > 0] = 255
        cv2.imshow('frame', im_toshow)
        k = cv2.waitKey(2) & 0xff
        if pre_rejects is None:
            while 1:
                k = cv2.waitKey(-1) & 0xff
                if k == ord('p'):
                    accepts.append(i)
                    print("ground truth for image", i, "passed and saved")
                    break
                elif k == ord('r'):
                    print("ground truth for image", i, "rejected")
                    rejects.append(i)
                    break
                elif k == ord('b') and i >= 1:
                    gts = gts[:-1]
                    images = images[:-1]
                    print("go back to frame", i-1)
                    if len(accepts) > 0 and accepts[-1] == i - 1:
                        accepts = accepts[:-1]
                    if len(rejects) > 0 and rejects[-1] == i - 1:
                        rejects = rejects[:-1]
                    i = i - 2
                    break
                else:
                    print("wrong input please input p for pass and r for reject b for backward")
        
            print("accepts", accepts)
            print("rejects", rejects)
        i = i + 1

    if pre_rejects is not None:
        rejects = pre_rejects
        for i in range(image_numbers):
            if i not in rejects:
                accepts.append(i)
    for i in range(image_numbers):
        np.save(gt_save_path + str(i) + '.npy', gts[i])
    for i in rejects:
        cv2.imwrite(image_save_path + str(i) + '.png', images[i])
    return accepts, rejects

def visualize_all(gt_path, img_path, image_numbers, accepts, rejects):
    #print("press any key to check accepts")
    #k = cv2.waitKey(-1) & 0xff
    #cv2.destroyAllWindows()
    for i in accepts:
        gt = np.load(gt_path + str(i) + '.npy')
        cv2.imshow("accepts" + str(i), gt.astype(np.uint8) * 255)
        k = cv2.waitKey(50) & 0xff
        cv2.destroyAllWindows()
    #print("press any key to check rejects")
    #k = cv2.waitKey(-1) & 0xff
    cv2.destroyAllWindows()
    for i in rejects:
        gt = np.load(gt_path + str(i) + '.npy')
        print(gt.shape)
        img = cv2.imread(img_path + str(i) + '.png')
        print(img_path + str(i) + '.png')
        print(img.shape)
        img[:,:,0][gt > 0] = 200
        cv2.imshow("rejects", img)
        k = cv2.waitKey(-1) & 0xff
    cv2.destroyAllWindows()

def replace_image(source_folder, target_folder):
    for i in range(1,7):
        for j in range(3):
            if i == 3 and j == 1:
                continue
            for side in ['left', 'right']:
                s = source_folder + str(i) + '/' +str(j) + '/'+ side
                t = target_folder + str(i) + '/' +str(j) + '/'+ side
                images = os.listdir(t)
                for img_path in images:
                    if img_path[-3:] == 'png':
                        shutil.copyfile(s + '/' + img_path, t + '/' + img_path)
                        print(s + '/' + img_path, t + '/' + img_path)





if __name__ == '__main__':
    s = 6
    v = 0
    side = 'right'
    gt1_path = 'dark/' 
    gt2_path = 'dark/'
    image_path = 'image/'
    gt_save_path = 'ground_truth/'
    img_save_path = 'mannual/'
    # accepts, rejects = combine_gts(gt1_path = gt1_path + str(s) + '/' + str(v) + '/' + side + '_sam_output/', gt2_path = gt2_path + str(s) + '/' + str(v) + '/' + side + '_sam_output/',
    #    image_path = image_path + str(s) + '/' + str(v) + '/' + side + '/',  gt_save_path = gt_save_path + str(s) + '/' + str(v) + '/' + side + '/', 
    #    image_save_path = img_save_path + str(s) + '/' + str(v) + '/' + side + '/', image_numbers = 250, pre_rejects = [31, 32, 49, 50, 60, 98, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 141, 193, 197])

    visualize_all(gt_path = gt_save_path + str(s) + '/' + str(v) + '/' + side + '/', img_path = img_save_path + str(s) + '/' + str(v) + '/' + side + '/',
        image_numbers = 300, accepts = range(300), rejects = [])

    # side = 'right'
    # accepts, rejects = combine_gts(gt1_path = gt1_path + str(s) + '/' + str(v) + '/' + side + '_sam_output/', gt2_path = gt2_path + str(s) + '/' + str(v) + '/' + side + '_sam_output/',
    #    image_path = image_path + str(s) + '/' + str(v) + '/' + side + '/',  gt_save_path = gt_save_path + str(s) + '/' + str(v) + '/' + side + '/', 
    #    image_save_path = img_save_path + str(s) + '/' + str(v) + '/' + side + '/', image_numbers = 250, pre_rejects = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 135, 139, 150, 154, 201, 217, 245])

    # visualize_all(gt_path = gt_save_path + str(s) + '/' + str(v) + '/' + side + '/', img_path = img_save_path + str(s) + '/' + str(v) + '/' + side + '/',
    #     image_numbers = 180, accepts = range(180), rejects = [])
    #background_subtraction(background_path = './'+str(s)+'/'+str(v)+'/dark/left/bg.png', image_path = './'+str(s)+'/'+str(v)+'/dark/left/', image_numbers=300, gt_save_path='./'+str(s)+'/'+str(v)+'/dark/left_prompt/')    
    #background_subtraction(background_path = './'+str(s)+'/'+str(v)+'/dark/right/bg.png', image_path = './'+str(s)+'/'+str(v)+'/dark/right/', image_numbers=300, gt_save_path='./'+str(s)+'/'+str(v)+'/dark/right_prompt/')
