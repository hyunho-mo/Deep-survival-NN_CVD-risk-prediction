import os
import zipfile
import numpy as np
import pandas as pd
import shutil
import stat
from tqdm import tqdm
import argparse

import nibabel as nib
from scipy import ndimage

import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, binary_erosion, area_opening, area_closing, binary_opening, binary_closing
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, chan_vese

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
import numpy as np
from skimage import measure
from scipy.ndimage import morphology
import models.fp_classifier as fp_classifier
import torch
import cv2


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))



def plot_slices_normalized_thres(num_rows, num_columns, width, height, data, filepath, label, thres_normal):
    """Plot a montage of 20 CT slices"""
    # data = np.rot90(np.array(data))
    data = np.transpose(data, (3, 2, 0, 1))
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            # image = simple_rotate(data[i][j])
            image = data[i][j][:, ::-1]
            # twod_image = data[i][j]
            twod_image = image
            sat_mask = twod_image > thres_normal

            true_positions = list(zip(*np.where(sat_mask == True)))
            for pos in true_positions:
                adjacent_list = []
                if (pos[0] >= 510) or (pos[1] >= 510):
                    pass
                else:
                    for x_pos in range(pos[0]-1, pos[0]+2):
                        for y_pos in range(pos[1]-1, pos[1]+2):
                            adjacent_list.append(twod_image[(x_pos, y_pos)]>= thres_normal)
                    if all(adjacent_list) == True:
                        pass
                    else:
                        sat_mask[pos] = 0

            imax2 = np.amax(twod_image)
            scale = 255.0/1
            data2 = (twod_image.astype(float) * scale).astype('uint8')
            data2_rgb = np.dstack((data2, data2, data2))
            data2_rgb[sat_mask] = np.array([255, 0, 0], np.uint8)

            axarr[i, j].imshow(twod_image, cmap="gray", vmin=0, vmax=1)
            axarr[i, j].imshow(data2_rgb, vmin=0, vmax=1)

            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    print ("check point")
    print ("filepath", filepath)
    plt.savefig(os.path.join(figure_dir, '%s_%s_thres.png' %(filepath, label)))





def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = np.numpy_function(scipy_rotate, [volume], np.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = np.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = np.expand_dims(volume, axis=3)
    return volume, label

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    # scan = scan.get_fdata()
    scan = scan.get_data()

    return scan

def normalize(volume):
    """Normalize the volume"""
    # check pixel min max first 
    # ranges from. -1024 HU to 3071 HU
    # min = -224
    # max = 1000
    hu_min = -256
    hu_max = 2048
    volume[volume < hu_min] = hu_min
    volume[volume > hu_max] = hu_max
    volume = ((volume - hu_min) / (hu_max - hu_min))

    # threshold value 
    thres = 130
    # print ("nomalized_thres", ((thres - min) / (thres - min)))

    # use "float16" if there's out of memory issue
    volume = volume.astype("float32")
    return volume

def resize_volume(img, resize):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = img.shape[-1]
    desired_width = resize
    desired_height = resize
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    print ("width_factor", width_factor)
    print ("width_factor", height_factor)
    print ("width_factor", depth_factor)
    print (img.dtype)
    # img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=3)
    return img


def simple_rotate(img):
    img = ndimage.rotate(img, -90, reshape=False)
    return img


def process_scan_normalize(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # print ("volume.shape", volume.shape)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = simple_rotate(volume)
    volume = np.expand_dims(volume, axis=3)
    return volume

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # print ("volume.shape", volume.shape)
    # Normalize
    # volume = normalize(volume)
    # Resize width, height and depth
    volume = simple_rotate(volume)
    volume = np.expand_dims(volume, axis=3)
    return volume


def count_slices(path):
    volume = read_nifti_file(path)  
    return volume.shape[2]


def protocol_thresholding_manual (heart_seg_images, thres_mask, thres):
    true_positions = list(zip(*np.where(thres_mask == True)))
    for pos in true_positions:

        if (pos[0] >= 510) or (pos[1] >= 510):
            pass
        else:

            adjacent_checker = heart_seg_images[pos[0]-1:pos[0]+1, pos[1]-1 : pos[1]+1] >= thres
            adjacent_checker_list = adjacent_checker.flatten().tolist()

            left_checker = heart_seg_images[pos[0]-2 : pos[0], pos[1]-1 : pos[1]+1] >= thres
            left_checker_list = left_checker.flatten().tolist()

            right_checker = heart_seg_images[pos[0] :pos[0]+2, pos[1]-1 : pos[1]+1] >= thres
            right_checker_list = right_checker.flatten().tolist()

            up_checker = heart_seg_images[pos[0]-1 : pos[0]+1, pos[1] : pos[1]+2] >= thres
            up_checker_list = up_checker.flatten().tolist()

            down_checker = heart_seg_images[pos[0]-1 : pos[0]+1, pos[1]-2 : pos[1]] >= thres
            down_checker_list = down_checker.flatten().tolist()


            if (all(adjacent_checker_list) == True) or (all(left_checker_list) == True) or (all(right_checker_list) == True) or (all(up_checker_list) == True) or (all(down_checker_list) == True):
                pass
            else:
                thres_mask[pos] = 0
    
    thres_images =  heart_seg_images * thres_mask
    # print ("thres_images.shape", thres_images.shape)

    return thres_images, thres_mask



def erosion_heart_mask (images):
    image_list = []
    for idx in range(images.shape[2]):
        slice = np.squeeze(images[:, :, idx])
        slice = morphology.binary_erosion(slice, iterations=10)
        image_list.append (slice)
    image_list = np.array(image_list)
    image_list = image_list.transpose(1, 2, 0)
    # print ("image_list.shape", image_list.shape)
        

    return image_list

################################################################################################

## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# configs_dir = os.path.join(current_dir, "configs")

# current working directory in c:
original_working_directory = os.getcwd()

figure_dir = os.path.join(current_dir, 'Figure_thres_128_crop_fp_calc')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)


figure_pid_dir = os.path.join(current_dir, 'Figure_score_compare')
if not os.path.exists(figure_pid_dir):
    os.makedirs(figure_pid_dir)

model_dir = os.path.join(current_dir, 'fp_detector')

# temp_image_dir = os.path.join(current_dir, 'temp_image') 
# print ("temp_image_dir", temp_image_dir)

temp_image_dir = "/data/scratch/hmo/ergo_heart"
storage_dir = "/data/scratch/hmo"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', type=str, help="pwd of nii file")
    parser.add_argument('-id', type=str, help="patient id")
    parser.add_argument('-raw', type=str, help="patient id", default="original")
    parser.add_argument('-heartslices', type=int, help="number of slices for pericardium", default=30)
    parser.add_argument('-resize', type=int, help="zoom", default=128)
    parser.add_argument('-patch_size', type=int, default=45, help="pwd of nii file")
    parser.add_argument('-px_thres', type=int, default=9, help="pwd of nii file")

    # "2021.nii"
    thres = 130
    # min = -224
    # max = 1000
    hu_min = -256
    hu_max = 2048
    thres_normal = ((thres - hu_min) / (hu_max - hu_min))
    thres_ori = 130
    # print ("thres_normal", thres_normal)

    pos_max = 511
    pos_min = 0

    pos_ref = 10

    ## Load fp detector (trained torch model) 
    if torch.cuda.is_available():
        model = torch.load(os.path.join(model_dir, 'fp_vgg_trained_model_all.pth'))
    else:
        model = torch.load(os.path.join(model_dir, 'fp_vgg_trained_model_all.pth'), map_location=torch.device('cpu'))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args = parser.parse_args()
    p_id = args.id
    patch_size = args.patch_size
    num_heartslice = args.heartslices
    resize = args.resize
    px_thres = args.px_thres

    # calc_dir_full = os.path.join(storage_dir, 'calc_array_%s_%s_crop_fp' %(num_heartslice, 512))
    # if not os.path.exists(calc_dir_full):
    #     os.makedirs(calc_dir_full)


    ## Loop (epifat pipeline) output directories and assign paths for nii files
    ids_dir_list = os.listdir(temp_image_dir)
    # print ("ids_dir_list", ids_dir_list)
    heart_num_slices_list = []

    df_score_comparison = pd.DataFrame([])
    patient_name_list = []
    agatston_score_list = []
    vol_score_list = []
    computed_agatston_list =[]
    computed_vol_list =[]

    tabular_dir = os.path.join(current_dir, 'Tabular')
    processed_csv_path =  os.path.join(tabular_dir, "RS_CT_subcohort_population_linked.csv")
    df_population = pd.read_csv(processed_csv_path, sep = ",", na_values=' ')   
    score_comparison_filepath = os.path.join(tabular_dir, 'score_comparison_px_vgg_%s_rev.csv' %px_thres)


    # print ("ids_dir_list", ids_dir_list)
    # ids_dir_list = ids_dir_list[:4]
    print ("len(ids_dir_list)", len(ids_dir_list))
    print ("ids_dir_list", ids_dir_list)
    for id_dir in tqdm(ids_dir_list):
        calc_masked_first_nii_img_list = []
        print ("id_dir", id_dir)
        
        # if int(id_dir) not in df_population["patient_name"]:
        #     continue

        ## Load agatston and volume scores
        df_ind = df_population[df_population["patient_name"]==int(id_dir)]
        if len(df_ind) == 0:
            continue
        agatston_score = df_ind["CAC"].values[0]
        vol_score = df_ind["CACvol"].values[0]

        print ("agatston_score", agatston_score)
        print ("vol_score", vol_score)
        
        # if (agatston_score == 0.0) and (vol_score == 0.0):
        #     continue

        patient_name_list.append(id_dir)
        agatston_score_list.append(agatston_score)
        vol_score_list.append(vol_score)



        # print ("len(calc_masked_first_nii_img_list)", len(calc_masked_first_nii_img_list))
        id_image_dir = os.path.join(temp_image_dir, id_dir)
        ori_filepath = os.path.join(id_image_dir, '%s_ct.nii.gz' %id_dir) 
        heartmask_filename = '%s_heartmask.nii.gz' %id_dir
        mask_filepath = os.path.join(id_image_dir, heartmask_filename) 
        if heartmask_filename not in os.listdir(id_image_dir):
            print ("missing pericaridum imgs for id: ", id_dir)
            continue
 
        # id_npy_filename = "calc_%s.npy" %id_dir
        # id_npy_filepath = os.path.join(calc_dir, id_npy_filename)

        # if id_npy_filename in os.listdir(calc_dir):
        #     print ("already processed")
        #     continue


        original_imgs = process_scan_normalize(ori_filepath) # Load original images 
        original_hu_imgs = process_scan(ori_filepath) # Load original images 
        mask_imgs = process_scan(mask_filepath) # Load heart segmentation masks
        mask_imgs = erosion_heart_mask (mask_imgs)  # Erosion to shrink (hyparams: iterations, default 10)
        mask_imgs = np.expand_dims(mask_imgs, axis=3)

        original_imgs = np.flip(original_imgs, axis=1)
        original_hu_imgs = np.flip(original_hu_imgs, axis=1)
        mask_imgs = np.flip(mask_imgs, axis=1)

        original_imgs = np.flip (original_imgs, axis= 2) # Inferior w.r.t slice index
        original_hu_imgs = np.flip (original_hu_imgs, axis= 2) # Inferior w.r.t slice index
        mask_imgs = np.flip (mask_imgs, axis= 2) # Inferior w.r.t slice index

        original_imgs = np.squeeze(original_imgs)
        original_hu_imgs = np.squeeze(original_hu_imgs)
        mask_imgs = np.squeeze(mask_imgs)
        # print ("mask_imgs.shape", mask_imgs.shape)
        ## Apply segmentation mask to extract heart from original image
        masked_imgs = []
        for slice_idx in range(original_imgs.shape[2]):
            test_image = np.squeeze(original_imgs[:, :, slice_idx])
            mask = np.squeeze(mask_imgs[:, :, slice_idx])

            masked_image = test_image * mask
            # print ("masked_image.shape", masked_image.shape)
            masked_imgs.append(masked_image)




        masked_imgs = np.array(masked_imgs)
        masked_imgs = masked_imgs.transpose(1, 2, 0)
        # images = np.expand_dims(masked_imgs, axis=3)
        # print ("masked_imgs.shape", masked_imgs.shape)
        ## check the number of slices for segmented heart (discard all false slices)
        heart_slice_count = 0
        true_index_list = []
        for slice_num in range(masked_imgs.shape[2]):
            current_slice = np.squeeze(masked_imgs[:, :, slice_num])
            # print ("np.all(current_slice == False)", np.all(current_slice == False))
            if not np.all(current_slice == False):
                heart_slice_count+=1
                true_index_list.append(slice_num)

        # print ("true_index_list", true_index_list)
        # print ("heart_slice_count", heart_slice_count)
        heart_num_slices_list.append(heart_slice_count)
        ori_heart_imgs = original_imgs[:,:,true_index_list[0]:true_index_list[-1]+1]
        ori_hu_heart_imgs = original_hu_imgs[:,:,true_index_list[0]:true_index_list[-1]+1]
        pericaridum_images = masked_imgs[:,:,true_index_list[0]:true_index_list[-1]+1]
        mask_imgs = mask_imgs[:,:,true_index_list[0]:true_index_list[-1]+1]
        # print ("pericaridum_images.shape", pericaridum_images.shape)

        # print ("ori_heart_imgs.shape", ori_heart_imgs.shape)
        ## if num of slices for segmented pericaridum is less than num_heartslice(input argument) then add dummy slices.
        if heart_slice_count <  num_heartslice:
            num_dummy_slices = num_heartslice - heart_slice_count
            dummy_array = np.zeros((pericaridum_images.shape[0], pericaridum_images.shape[1], num_dummy_slices))
            # print ("dummy_array.shape", dummy_array.shape)
            ori_heart_imgs = np.concatenate([ori_heart_imgs, dummy_array], axis=2)
            ori_hu_heart_imgs = np.concatenate([ori_hu_heart_imgs, dummy_array], axis=2)
            pericaridum_images = np.concatenate([pericaridum_images, dummy_array], axis=2)
            mask_imgs = np.concatenate([mask_imgs, dummy_array], axis=2)
            # print ("pericaridum_images.shape", pericaridum_images.shape)

        ## Select first num_heartslice 
        pericaridum_images = pericaridum_images[:,:,0:num_heartslice]
        ori_heart_imgs = ori_heart_imgs[:,:,0:num_heartslice]
        ori_hu_heart_imgs = ori_hu_heart_imgs[:,:,0:num_heartslice]
        mask_imgs = mask_imgs[:,:,0:num_heartslice]
        # print ("pericaridum_images.shape", pericaridum_images.shape)

        ## Plot pericardium with thresholding
        # pericaridum_images = pericaridum_images.astype(np.float16)
        # plot_slices_normalized_thres(int(pericaridum_images.shape[2]/10), 10, pericaridum_images.shape[0], pericaridum_images.shape[1], np.expand_dims(pericaridum_images, axis=3), id_dir, "ori", thres_normal)


        computed_agatston_score = 0
        computed_vol_score = 0

        for slice_idx in range(pericaridum_images.shape[2]):

            patch_slice_list = []
            label_slice_list = []
            lesion_size_list = []


            first_nii_img = ori_heart_imgs[:,:, slice_idx]
            ori_hu_nii_img = ori_hu_heart_imgs[:,:, slice_idx]
            masked_first_nii_img = pericaridum_images[:,:, slice_idx]
            first_heart_mask = mask_imgs[:,:, slice_idx]


            ## Thresholding pericardium
            calc_mask = first_heart_mask > thres_normal
            # calc_masked_first_nii_img, _ = protocol_thresholding_manual (masked_first_nii_img, calc_mask, thres_normal)
            # calc_masked_first_nii_img_raw, calc_mask = protocol_thresholding_manual (masked_first_nii_img, calc_mask, thres_normal)

            calc_masked_first_nii_img, calc_mask = protocol_thresholding_manual (masked_first_nii_img, calc_mask, thres_normal)
            calc_masked_first_nii_img_raw = calc_masked_first_nii_img.copy()

            if slice_idx == 0:
                slide_ref = true_index_list[slice_idx]
            else:
                slide_ref += 1

            ## Edge detection and findcontours
            # Find the contours on the inverted binary image, and store them in a list
            # Contours are drawn around white blobs.
            # hierarchy variable contains info on the relationship between the contours
            contours, hierarchy = cv2.findContours(calc_mask.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
            ## Translate contours
            contours_trans_list = []
            for index, c in enumerate(contours):
                contours_trans_list.append(c - 1)

            contours = tuple(contours_trans_list)

            # # Draw the contours (in red) on the original image and display the result
            # # Input color code is in BGR (blue, green, red) format
            # # -1 means to draw all contours    
            # gray_three = cv2.merge([first_nii_img_wo_norm,first_nii_img_wo_norm,first_nii_img_wo_norm])
            gray_three = cv2.merge([first_nii_img,first_nii_img,first_nii_img])
            with_contours = cv2.drawContours(gray_three, contours, -1 ,(255,0,0), 1)


            calc_postivie_pos_list = []
            # calc_masked_first_nii_img_raw = calc_masked_first_nii_img

            # if len(contours) >= 10 :
            #     px_thres = 36
            # else:
            #     px_thres = 16

            

            if slice_idx == 0:
                prev_slice_tp_x_list = []
                prev_slice_tp_y_list = []
                current_slice_tp_x_list = []
                current_slice_tp_y_list = []
            elif slice_idx >= 1:
                prev_slice_tp_x_list = current_slice_tp_x_list
                prev_slice_tp_y_list = current_slice_tp_y_list
                current_slice_tp_x_list = []
                current_slice_tp_y_list = []

            ### Loop for contours ###
            for index, c in enumerate(contours):
                x, y, w, h = cv2.boundingRect(c)
                # # Make sure contour area is large enough
                # cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,0), 0)

                x_center = x + int(w/2)
                y_center = y + int(h/2)

                new_x = x_center - int(patch_size/2)
                new_y = y_center - int(patch_size/2)
            
                pos_match_counter = 0
                

                ## Crop image based on it coordinates in opencv
                ## cv2.rectangle(rightImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                ## cropImg=rightImg[y:y+h,x:x+w]
                img_patch = first_nii_img[new_y:new_y+patch_size, new_x:new_x+patch_size]
                # print ("first_nii_img.shape", first_nii_img.shape)
                # print ("new_y", new_y)
                # print ("new_x", new_x)

                # print ("new_y, new_y+patch_size", new_y, new_y+patch_size)
                # print ("new_x, new_x+patch_size", new_x,  new_x+patch_size)

                ## Prepare 2d patch conveying spational info in y-direction 
                y_pos_vector = np.arange(new_y, new_y+patch_size)
                # print ("y_pos_vector", y_pos_vector)
                y_pos_patch = np.tile(y_pos_vector, (patch_size, 1))
                y_pos_patch = np.transpose(y_pos_patch)
                # print ("y_pos_patch", y_pos_patch)

                ## Prepare 2d patch conveying spational info in x-direction 
                x_pos_vector = np.arange(new_x, new_x+patch_size)
                x_pos_patch = np.tile(x_pos_vector, (patch_size, 1))
                # print ("x_pos_patch", x_pos_patch)

                ## Normalize x, y pos patch
                x_pos_patch = (x_pos_patch-pos_min)/(pos_max-pos_min)
                y_pos_patch = (y_pos_patch-pos_min)/(pos_max-pos_min)

                # ## Prepare 2d patch conveying the slice index
                # sindex_vector = np.ones((patch_size,), dtype=int)          
                # sindex_vector = sindex_vector * slide_ref  
                # sindex_patch = np.tile(sindex_vector, (patch_size, 1))
                # # print ("sindex_patch", sindex_patch)

                ## Stack patch image and x,y spational information. Each of them processed in the different channel of the network.
                # patch = np.dstack((img_patch, x_pos_patch, y_pos_patch, sindex_patch)) 

                if img_patch.shape[0] == 0:
                    continue       
                else:
                    
                    ## neglects contours on the corners 
                    if (img_patch.shape[0] != 45) or (img_patch.shape[1] != 45):
                        print ("neglect contours on the corners")
                        continue

 
                    patch = np.dstack((img_patch, x_pos_patch, y_pos_patch))        

                ## Check annotations and compare it for labeling the patch
                # Retrieve pixel positions of the pixels in the contour
                mask = np.zeros(first_nii_img.shape, np.uint8)
                mask = cv2.merge([mask,mask,mask])
                # cv2.drawContours(mask, annot_contours, -1 ,(255,0,0), thickness=cv2.FILLED)
                # cv2.drawContours(mask, annot_contours, -1 ,(255,0,0), 1)

                mask2 = np.zeros(first_nii_img.shape, np.uint8)
                mask2 = cv2.merge([mask2,mask2,mask2])




                cv2.drawContours(image=mask, contours=contours, contourIdx=index, color=(0,255,255), thickness=cv2.FILLED)
                # cv2.drawContours(image=mask, contours=contours, contourIdx=index, color=(0,255,255), thickness=-1)

                cv2.drawContours(image=mask2, contours=contours, contourIdx=index, color=(0,255,255), thickness=1)

                # fig = plt.figure()
                # # plt.imshow(masked_first_nii_img, cmap=plt.cm.gray)
                # plt.imshow(mask2, cmap="gray", vmin=0)
                # # plt.savefig(os.path.join(figure_pid_dir, '%s_%s_nifti_calc.png' %(id_dir, slide_ref)))
                # plt.savefig(os.path.join(figure_pid_dir, '%s_%s_mask2_%s.png' %(id_dir, slice_idx, index)))


                mask_array = np.sum(mask, axis=2)
                mask2_array = np.sum(mask2, axis=2)

                calc_candid_pos = np.stack(np.nonzero(mask_array), axis=-1)
                calc_candid_pos = unique_rows(calc_candid_pos)
                calc_contour_pos = np.stack(np.nonzero(mask2_array), axis=-1)
                calc_contour_pos = unique_rows(calc_contour_pos)
                calc_neglect_pos = np.concatenate((calc_candid_pos, calc_contour_pos))
                calc_neglect_pos = unique_rows(calc_neglect_pos)



                calc_candid_pos_list = []
                calc_pos_y = calc_candid_pos[:,0]
                calc_pos_x = calc_candid_pos[:,1]
                calc_candid_value_list = []
                for y, x in zip(calc_pos_y, calc_pos_x):
                    # calc_candid_pos_list.append((x,y))
                    calc_candid_pos_list.append((x,y))

                    value = calc_masked_first_nii_img[y,x]
                    calc_candid_value_list.append(value)
                    # print ("value", value)
               

                # print ("calc_candid_pos_list", calc_candid_pos_list)
                # print ("num of px: ", len(calc_candid_pos_list))
                    
                # print ("len(calc_candid_pos)", len(calc_candid_pos))
                # print ("len(calc_candid_value_list)", len(calc_candid_value_list))
                # print ("calc_candid_value_list", calc_candid_value_list)
                # print ("thres_normal", thres_normal)

                above_thres_list = [num for num in calc_candid_value_list if num >= thres_normal]
                # print ("len(above_thres_list)", len(above_thres_list))
                # print ("above_thres_list", above_thres_list)



                if len(above_thres_list) <= px_thres:
                    # print ("too small lension")
                    label = 0
                    # print ("calc_candid_pos", calc_candid_pos)
                    calc_masked_first_nii_img[calc_neglect_pos] = 0
                    lesion_size = "small"
                    continue

                else:
                    lesion_size = "large"
                    ## check the output of DL-based fp detector when the size of lesion is larger than 6 pixels        
                    model.eval()
                    with torch.no_grad():
                        patch_4d = np.expand_dims(patch, axis=0)
                        # print ("patch_4d.shape", patch_4d.shape)
                        img = np.transpose(patch_4d, axes=[0, 3, 2, 1])
                        img = torch.Tensor(img)
                        # print ("img.shape", img.shape)
                        img = img.to(device)
                        output = model(img)
                        exp = torch.exp(output).cpu()
                        exp_sum = torch.sum(exp, dim=1) 
                        softmax = exp/exp_sum.unsqueeze(-1)
                        prob = list(softmax.detach().cpu().numpy())
                        predictions = np.argmax(prob, axis=1)
                        # print ("output", output)
                        # print ("softmax", softmax)
                        # print ("prob", prob)
                        # print ("predictions", predictions)
                        # print ("predictions.item()", predictions.item())



                    # label = 1
                    label = int(predictions.item())
                    # for calc_annot_pos in calc_annot_pos_all_list:
                    #     # print ("calc_annot_pos", calc_annot_pos)
                    


                    if label == 1: 
                        ## Calculate area of this contour

                        # area_px = cv2.contourArea(c)
                        ## Number of pixels
                        # area_px = len(calc_neglect_pos)
                        area_px = len(calc_candid_pos)


                        # print ("area_px", area_px)
                        area_mm = area_px*(180/512)*(180/512)  
                        # area_mm = area_px*(1/3)*(1/3)  

                        # print ("area_mm", area_mm)
                        ## max HU value within contour
                        # print ("calc_masked_first_nii_img.shape", calc_masked_first_nii_img.shape)
                        # print ("calc_neglect_pos.shape", calc_neglect_pos.shape)

                        calc_candid_pos_list = []
                        calc_pos_y = calc_candid_pos[:,0]
                        calc_pos_x = calc_candid_pos[:,1]
                        # calc_pos_y = calc_neglect_pos[:,0]
                        # calc_pos_x = calc_neglect_pos[:,1]
                        hu_values = []
                        for y, x in zip(calc_pos_y, calc_pos_x):
                            # calc_candid_pos_list.append((x,y))
                            calc_candid_pos_list.append((y,x))
                            hu_value = int(ori_hu_nii_img[(y,x)]) 
                            hu_values.append(hu_value)

                        # hu_values = [calc_masked_first_nii_img[px] for px in calc_candid_pos_list]
                        # print ("hu_values", hu_values)
                        max_hu = max(hu_values)
                        # print ("max_hu", max_hu)

                        if 130 <= max_hu <= 199:
                            hu_factor = 1
                        elif 200 <= max_hu <= 299:
                            hu_factor = 2
                        elif 300 <= max_hu <= 399:
                            hu_factor = 3
                        elif 400 <= max_hu:
                            hu_factor = 4                        
                        # Factor for agatston
                        # print ("hu_factor", hu_factor)
                        
                        contour_agatston_score = area_mm * hu_factor

                        computed_agatston_score += contour_agatston_score
                        computed_vol_score += area_mm * 3
                        # computed_vol_score += area_mm * 2



            # plt.imshow(with_contours)
            # plt.savefig(os.path.join(figure_pid_dir, '%s_%s_contours.png' %(id_dir, slice_idx)))
        computed_agatston_score = round(computed_agatston_score, 1)
        computed_vol_score = round(computed_vol_score, 1)
        print ("computed_agatston_score", computed_agatston_score) 
        print ("computed_vol_score", computed_vol_score) 
        computed_agatston_list.append(computed_agatston_score)
        computed_vol_list.append(computed_vol_score)







    df_score_comparison["patient_name"] = patient_name_list
    df_score_comparison["given_agatston"] = agatston_score_list
    df_score_comparison["computed_agatston"] = computed_agatston_list
    df_score_comparison["given_vol"] = vol_score_list
    df_score_comparison["computed_vol"] = computed_vol_list
    df_score_comparison["agatston_diff"] = (df_score_comparison['computed_agatston'] - df_score_comparison['given_agatston']).abs()
    df_score_comparison["vol_diff"] = (df_score_comparison['computed_vol'] - df_score_comparison['given_vol']).abs()

    agatston_diff_avg = df_score_comparison.loc[:, 'agatston_diff'].mean()
    vol_diff_avg = df_score_comparison.loc[:, 'vol_diff'].mean()
    
    print ("agatston_diff_avg", agatston_diff_avg)
    print ("vol_diff_avg", vol_diff_avg)

    df_score_comparison.to_csv(score_comparison_filepath, sep = ",", index = False)


if __name__ == '__main__':
    main()    