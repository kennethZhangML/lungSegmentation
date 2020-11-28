def read_nii(file_path):
    ct_scan = nib.load(file_path)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)