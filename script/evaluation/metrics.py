import numpy as np
import cv2 as cv

def abs_rel(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    absrel { float }
        A float representing the absolute relative error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of AbsRel due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Prevent any divide by zero errors
    predicted += 1
    ground += 1
    
    # Calculate the absolute relative error
    absmap = np.abs(predicted - ground)
    absmap[~valid_mask] = 0.0
    absmap = absmap / ground
    absrel = np.sum(absmap) / np.sum(valid_mask) 

    return absrel

def rmse(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.

    Output:
    rmse { float }
        A float representing the root mean square error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of RMSE due to inconsistent dimensions between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Calculate the root mean square error
    error = predicted - ground
    error[~valid_mask] = 0.0
    rmse = (np.sum(error ** 2) / np.sum(valid_mask)) ** 0.5

    return rmse

def dbe_accuracy(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.
    
    Output:
    dbe_acc { float }
        A float representing the depth boundary accuracy error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of dbe_accuracy due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None

    # Requires edge maps
    predicted_edges = cv.Canny(predicted, 100, 200)
    ground_edges = cv.Canny(ground, 100, 200)

    # Rescale predicted_edges so it is between 0 and 1
    predicted_edges = predicted_edges / 255
 
    # Euclidean Distance Transform
    ground_dist = cv.distanceTransform(ground_edges, cv.DIST_L2, 5)

    # Apply valid mask.
    predicted_edges *= valid_mask
    ground_dist *= valid_mask

    #print(f"Predicted Edges:\n{predicted_edges}\n\nGround EDT:\n{ground_dist}")

    # Calculate depth boundary accuracy error
    dbe_acc = np.sum(ground_dist * predicted_edges) / np.sum(predicted_edges)
    return dbe_acc

def dbe_completeness(predicted: np.ndarray, ground: np.ndarray, valid_mask: np.ndarray):
    """
    Parameters:
    predicted { np.array(ndim=2) } 
        A 2D numpy array containing a predicted depth map. This is generally the result from applying a depth estimation model.
    ground { np.array(ndim=2) } 
        A 2D numpy array containing the ground truth depth map. Should be representing the same image used in prediction.
    
    Output:
    dbe_comp { float }
        A float representing the depth boundary completeness error of the data.
        Returns None if the array dimensions of `predicted` does not match `ground`.
    """

    # Ensure predicted and ground have matching dimensions.
    if predicted.shape != ground.shape:
        print("WARNING: Skipped evaluation of dbe_completeness due to inconsistent sizes between predicted and ground truth values.\n"
              f"{predicted.shape} in predicted, {ground.shape} in ground truth.")
        return None
    
    # Calculate the depth bounary completeness error by calling the accuracy error function with reversed arguments 

    # Normalize to 0–255, because canny maps work on 8bit grayscale imgs.
    predicted = predicted - predicted.min()
    if predicted.max() > 0:
        predicted = predicted / predicted.max()
    predicted = (predicted * 255).astype(np.uint8)

    ground = ground - ground.min()
    if ground.max() > 0:
        ground = ground / ground.max()
    ground = (ground * 255).astype(np.uint8)


    dbe_comp = dbe_accuracy(ground, predicted, valid_mask)

    return dbe_comp

