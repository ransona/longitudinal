import organise_paths
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def compare_sessions(userID,expIDs):
    # specify which sessions to compare
    # userID = 'pmateosaparicio'
    # expIDs = ['2023-05-30_03_ESMT126', '2023-05-30_04_ESMT126','2023-05-30_05_ESMT126','2023-05-30_06_ESMT126','2023-05-30_07_ESMT126']

    # 
    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expIDs[1])
    # find number of depths
    dir_path = os.path.join(exp_dir_processed,'suite2p')
    plane_folders = sorted([name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name)) and 'plane' in name])
    plane_count = len(plane_folders)
    # make depth x exps subplot
    fig, axs = plt.subplots(plane_count,len(expIDs), sharex=True, sharey=True)
    # cycle through exps and depths displaying enhanced image of each
    # register each depth to the first session

    # registration params etc

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations and termination epsilon
    number_of_iterations = 500
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    ref_images = {}
    for iExp in range(len(expIDs)):
        print('Starting experiment ' + str(iExp) + ' of ' + str(len(expIDs)))
        animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expIDs[iExp])
        for iPlane in range(plane_count):
            print('Starting plane ' + str(iPlane) + ' of ' + str(plane_count))
            plane_path = os.path.join(exp_dir_processed,'suite2p','plane' + str(iPlane))
            # load the np stat file 
            s2p_stat = np.load(os.path.join(plane_path, 'stat.npy'), allow_pickle=True)
            s2p_ops = np.load(os.path.join(plane_path, 'ops.npy'), allow_pickle=True).item()
            if iExp == 0:
                # then store the images from each depth so that images from other experiments
                # can be aligned to it
                ref_images[iPlane] = s2p_ops['meanImgE']
                axs[iPlane,iExp].imshow(s2p_ops['meanImgE'],cmap='gray')
                axs[iPlane,iExp].set_ylabel('Plane ' + str(iPlane))
            else:
                # align the image to the one from the first experiment
                # Run the ECC algorithm to align the images
                img1 = ref_images[iPlane] # reference image
                img2 = s2p_ops['meanImgE'] # current plane
                # Run the ECC algorithm to align the images
                (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    img_aligned = cv2.warpPerspective(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:
                    img_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                axs[iPlane,iExp].imshow(img_aligned,cmap='gray')
            if iPlane == 0:
                # add label of experiment
                axs[iPlane,iExp].set_title('Exp ' + str(iExp))
            # axs[iPlane,iExp].axis('on')
            axs[iPlane,iExp].xaxis.set_ticks([])
            axs[iPlane,iExp].yaxis.set_ticks([])

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)
    print('Waiting for plot window to be closed...')
    plt.show(block=True)
    
    # x = 0

if __name__ == "__main__":
    compare_sessions()