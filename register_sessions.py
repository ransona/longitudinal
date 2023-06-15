import organise_paths
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import util
from PIL import Image
import copy
import pickle

# algorithm
# 1)    calculate transform required to register each plane of each experiment to the first experiment
# 2)    put each valid roi into an roi 'image', filling in the outer pixels so it isn't a hollow ring
# 2.1)  if any roi overlaps with an existing one nullify overlapping pixels
# 3)    warp these images based on 1)
# 4)    cycle through all roi IDs in exp0 checking for their overlap with the other experiments
# 5)    if there is overlap > x% record these as potential longitudinal matches and remove pixels of the associated rois from exp1->expn
# 5.1)  for rois with no valid overlap store them as 'one off' cells (i.e. only appearing active in session 0)
# 6)    cycle through all roi IDs in exp1 checking for their overlap with the other experiments (except the first)
# 7)    repeat until all cells in all experiments are accounted for



def register_sessions(userID,expIDs):
    # specify which sessions to compare
    userID = 'pmateosaparicio'
    expIDs = ['2023-05-30_03_ESMT126', '2023-05-30_05_ESMT126','2023-05-30_06_ESMT126','2023-05-30_07_ESMT126']
    # expIDs = ['2023-05-30_03_ESMT126', '2023-05-30_05_ESMT126']

    # 
    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expIDs[0])
    # find number of depths
    dir_path = os.path.join(exp_dir_processed,'suite2p')
    plane_folders = sorted([name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name)) and 'plane' in name])
    plane_count = len(plane_folders)
    # make depth x exps subplot
    #fig, axs = plt.subplots(plane_count,len(expIDs), sharex=True, sharey=True)
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

    all_cc = {}
    all_warp_matrix = {}
    all_mask_composites = {}
    all_mask_contours = {}
    all_fov_aligned = {}
    

    for iExp in range(len(expIDs)):
        print('Starting experiment ' + str(iExp) + ' of ' + str(len(expIDs)))
        animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expIDs[iExp])
        all_cc[iExp] = {}
        all_warp_matrix[iExp] = {}
        all_mask_composites[iExp] = {}
        all_mask_contours[iExp] = {}
        all_fov_aligned[iExp] = {}


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

            # align the image to the one from the first experiment
            # Run the ECC algorithm to align the images
            img1 = ref_images[iPlane] # reference image
            img2 = s2p_ops['meanImgE'] # current plane
            # Run the ECC algorithm to align the images
            (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)
            # store the cc and warp_matrix for the plane
            all_cc[iExp][iPlane] = cc 
            all_warp_matrix[iExp][iPlane] = warp_matrix
            all_mask_contours[iExp][iPlane] = {}
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                img_aligned = cv2.warpPerspective(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                img_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            all_fov_aligned[iExp][iPlane] = img_aligned

            # axs[iPlane,iExp].imshow(img_aligned,cmap='gray')
            # if iPlane == 0:
            #     # add label of experiment
            #     axs[iPlane,iExp].set_title('Exp ' + str(iExp))
            #     # axs[iPlane,iExp].axis('on')
            #     axs[iPlane,iExp].xaxis.set_ticks([])
            #     axs[iPlane,iExp].yaxis.set_ticks([]) 

            # generate roi mask composite for this experiment / depth
            mask_composite = np.zeros(img1.shape)
            # iterate through all valid roi IDs
            s2p_iscell = np.load(os.path.join(plane_path, 'iscell.npy'), allow_pickle=True)
            valid_cell_ids = np.where(s2p_iscell[:,0]==1)
            valid_cell_ids = valid_cell_ids[0]
            for iRoi in valid_cell_ids:
                xpix = (s2p_stat[iRoi]['xpix'])
                ypix = (s2p_stat[iRoi]['ypix'])
                empty_map = np.zeros(img1.shape).astype(np.uint8)
                empty_map[ypix,xpix] = 255
                # Define a kernel for the dilation and erosion operations
                kernel = np.ones((5,5),np.uint8)
                # Dilate the image to close any gap in the ROI
                dilated = cv2.dilate(empty_map, kernel, iterations = 1)
                # Fill the holes in the dilated ROI
                _, thresh = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_mask_contours[iExp][iPlane][iRoi] = contours[0]
                filled = cv2.drawContours(dilated, contours, -1, (255,0,0), thickness=cv2.FILLED)
                # Erode the filled image to restore the original size of the ring
                final_roi = cv2.erode(filled, kernel, iterations = 1)
                mask_composite[final_roi==255] = iRoi

            # warp the roi map
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                mask_composite_aligned = cv2.warpPerspective(mask_composite, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
            else:
                mask_composite_aligned = cv2.warpAffine(mask_composite, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
            
            all_mask_composites[iExp][iPlane] = mask_composite_aligned
            # util.imshow(mask_composite_aligned)
            # axs[iPlane,iExp].imshow(mask_composite_aligned,cmap='jet')

    all_mask_composites_working = copy.deepcopy(all_mask_composites)
    # make space to store matches and overlap
    all_matches = {}
    all_overlaps = {}  
    all_ref_roi_crop = {}
    all_roi_outlines = {} # stores outlines of matched rois for later plotting over roi images

    for iPlane in range(plane_count):
        all_matches[iPlane] = {}
        all_overlaps[iPlane] = {}
        all_ref_roi_crop[iPlane] = {}
        all_roi_outlines[iPlane] = {}
        total_matches_in_plane = -1
        for iExpRef in range(len(expIDs)):
            # cycle through all ROIs in each experiment
            unique_ids = np.unique(all_mask_composites_working[iExpRef][iPlane])
            unique_ids = unique_ids[unique_ids!=0]
            for iRoi in range(len(unique_ids)):
                # for each experiment detect the most common non-zero pixel in the footprint of the 
                # reference experiments ROIs and compute the % overlap of the reference ROI with this
                # most common roi
                # make space to store roi IDs similar to the current reference ROI
                best_match_id = np.zeros([1,len(expIDs)]) # most overlapping from each experiment
                overlap = np.zeros([1,len(expIDs)]) # how much overlap (fractional)
                # find the edges of the ref roi that will later be used for displaying it
                roi_pix_coordinates = np.where(all_mask_composites_working[iExpRef][iPlane]==unique_ids[iRoi])
                ref_roi_crop = {}
                roi_outlines = {}
                # padding around cell image
                cell_img_padding = 10 # pix
                frame_size = all_fov_aligned[0][0].shape
                ref_roi_crop['top']    = max(min(roi_pix_coordinates[0]) - cell_img_padding,0)
                ref_roi_crop['bottom'] = min(max(roi_pix_coordinates[0]) + cell_img_padding,frame_size[0]-1)
                ref_roi_crop['left']   = max(min(roi_pix_coordinates[1]) - cell_img_padding, 0)
                ref_roi_crop['right']  = min(max(roi_pix_coordinates[1]) + cell_img_padding,frame_size[1]-1)
                # find pix of reference experiment
                ref_roi_pix = np.where(all_mask_composites_working[iExpRef][iPlane]==unique_ids[iRoi])
                # convert to linear index
                # ref_roi_pix = np.ravel_multi_index(ref_roi_pix,all_mask_composites[iExpRef][iPlane].shape)
                for iExp in range(len(expIDs)):
                    # util.imshowblend(all_mask_composites[iExpRef][iPlane], all_mask_composites[iExp][iPlane])
                    # find roi indices within reference roi in new experiment
                    rois_in_ref_roi = all_mask_composites_working[iExp][iPlane][ref_roi_pix[0],ref_roi_pix[1]]
                    # remove zero values
                    rois_in_ref_roi = rois_in_ref_roi[rois_in_ref_roi != 0]

                    if rois_in_ref_roi.size > 0:
                        # find which roi is most common within ref rois pixels
                        counter = Counter(rois_in_ref_roi)
                        most_common = counter.most_common(1)
                        most_common = most_common[0][0]
                        # calculate what fraction of the total unique pixels which make up the two rois 
                        # are overlapping
                        total_pix_count = len(ref_roi_pix[0]) + len(np.where(all_mask_composites_working[iExp][iPlane]==most_common)[0]) - len(np.where(rois_in_ref_roi == most_common)[0])
                        overlap_pix_count = len(np.where(rois_in_ref_roi == most_common)[0])
                        overlap_frac = overlap_pix_count / total_pix_count
                        best_match_id[0,iExp] = most_common
                        overlap[0,iExp] = overlap_frac
                        # set pixels of matched roi from iExp to zero so it doesn't get used as 
                        # a seed in the future iterations and then get double counted
                        all_mask_composites_working[iExp][iPlane][np.where(all_mask_composites_working[iExp][iPlane]==most_common)] = 0
                    else:
                        best_match_id[0,iExp] = np.nan
                        overlap[0,iExp] = np.nan

                # add the found matched cells as a new row
                total_matches_in_plane = total_matches_in_plane  + 1
                all_matches[iPlane][total_matches_in_plane] = best_match_id[0]
                all_overlaps[iPlane][total_matches_in_plane] = overlap[0]
                all_ref_roi_crop[iPlane][total_matches_in_plane] = ref_roi_crop

                # set pixels of ref experiment to zero so this roi doesn't get detected as overlapping
                # with other rois from other experiments
                all_mask_composites_working[iExpRef][iPlane][ref_roi_pix[0],ref_roi_pix[1]] = 0
    
    # we now have all cells
    # make structure for bringing together all of the data
    match_data = {}
    match_data['all_fov_aligned']       = all_fov_aligned
    match_data['all_mask_composites']   = all_mask_composites
    match_data['all_matches']           = all_matches
    match_data['all_overlaps']          = all_overlaps
    match_data['all_ref_roi_crop']      = all_ref_roi_crop
    match_data['all_warp_matrix']       = all_warp_matrix

    # save the data on tracked rois to first experiment in sequence
    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expIDs[0])    
    long_path = os.path.join(exp_dir_processed,'long')
    os.makedirs(long_path, exist_ok=True)
    print('Saving data to ' + os.path.join(long_path,'long.pickle'))
    with open(os.path.join(long_path,'long.pickle'), 'wb') as pickle_out:
        pickle.dump(match_data, pickle_out)
   

    # Create the subplots
    # fig, axes = plt.subplots(1, len(expIDs), sharex=True, sharey=True)
    # plt.ion()
    # plt.show()
    
    # # check matches
    # for iPlane in range(plane_count):
    #     for iLongRoi in range(len(all_matches[iPlane])):
    #         for iRoi in range(len(expIDs)):
    #             # cycle through each experiment
    #             cell_ID = all_matches[iPlane][iLongRoi][iRoi]
    #             axes[iRoi].clear()
    #             axes[iRoi].imshow(all_fov_aligned[iRoi][iPlane], origin='upper')
    #             if not np.isnan(cell_ID):
    #                 roi_mask = np.zeros(all_mask_composites[iExpRef][iPlane].shape)
    #                 roi_mask = all_mask_composites[iRoi][iPlane] == cell_ID
    #                 roi_mask = roi_mask.astype('uint8') * 255
    #                 _, thresh = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    #                 mask_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #                 mask_contours = mask_contours[0]
    #                 xvals = np.append(mask_contours[:,0,0],mask_contours[0,0,0])
    #                 yvals = np.append(mask_contours[:,0,1],mask_contours[0,0,1])
    #                 axes[iRoi].plot(xvals,yvals,'r')
    #                 # axes[iRoi].axis('off')
    #         # zoom in to region of the roi in question
    #         plt.xlim(all_ref_roi_crop[iPlane][iLongRoi]['left'],all_ref_roi_crop[iPlane][iLongRoi]['right'])
    #         plt.ylim(all_ref_roi_crop[iPlane][iLongRoi]['bottom'],all_ref_roi_crop[iPlane][iLongRoi]['top'])


if __name__ == "__main__":
    register_sessions('x','y')