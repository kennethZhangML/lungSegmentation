# Lung Segmentation and COVID-19 Infection Prediction
Kenneth's Lung Segmentation and COVID-19 Infection Prediction Program for HackTable2020

## Inspiration
Since the beginning of the COVID-19 pandemic, we have utilized traditional methods of COVID-19 diagnosis and detection. While utilizing trained professionals and experts in medicine to assess a patient is relatively effective, it is not efficient. By training a UNet, a neural network designed for biomedical imaging and segmentation, we can perhaps generate more precise, effective, and efficient methods of detection of COVID-19. 

## What it does
The following project uses a UNet, a neural network designed for biomedical imaging, classification, and segmentation, to detect and segment the specific parameters and perimeters of the lung in an NII/CT scan. It is able to recognize the precise locations of a lung in an image and is also able to segment and predict where the infection is. 

## How I built it
The UNet can be trained just like any other neural network in the plethora of neural network architectures available to us as Machine Learning engineers. In addition, the dataset provided was also preprocessed and organized so training would be more organized and accessible at once.

1. Preprocessing + Reading the Lung Masks, Segmentation Masks and Infection Masks
In the first stages, we must read the masks in the correct manner. Since they were ".nii" files, a file dedicated to neuroimaging, we must convert each image to a NumPy array in a certain fashion. Here is the following code:

```python
def read_nii(file_path):
    ct_scan = nib.load(file_path)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)
```
As you can see, the first line defines the function. The second line loads the ".nii" file from the file path. The array variable is set and gets the NumPy array of the image while rotating it 90 degrees.

2. Visualizing the Images (Masks + CT_Scans)
We then can visualize the masks and the CT scans together. By simply using MatPlotLib, we can plot each of the images and see the masks that will train the UNet to recognize the precise locations of the lungs. The infection masks will allow the UNet to be trained to recognize the probable locations of COVID-19 infections in the lung masks.
The lung and infection masks are of CT scans that have been confirmed to contain COVID-19 in them, and are able to plot them so that we can visualize where the probable locations of COVID-19 could be.

3. Training the UNet 
We must first define the UNet, either through loading pre-trained weights or by completely implementing it from the ground up. We can then fit the model to the masks and scans. The model will loop through the entire dataset around 10 times in order to maximize the performance and accuracy of the model. The model will then be validated on our provided validation set with the same image shapes. 

4. Visualizing the Training Results
It is important that all models are evaluated at the end of the process. We use the Matplotlib library to plot the Training and validation loss, as well as the training and validation accuracy of the model over the 10 epochs. I completed steps 1 - 4 around 5 times in order to visualize some consistency. The model was able to consistently maintain over a 99% accuracy using the same number of epochs. 

## Challenges I ran into
While training the UNet was rather simple, loading the data in and converting the ".nii" files into NumPy arrays was difficult. I was not previously equipped with an understanding of the format of the images in the dataset. The first attempt I made to convert the images to a NumPy array resulted in me disrupting and causing detrimental damage to the dataset, as I had accidentally manipulated some of the images. On my second attempt, I was able to successfully import and load the image and convert them into NumPy arrays via the read_nii.py function. 

note: If you want the read_nii.py function to work, please import the function in the main.ipynb file. Make sure the read_nii.py file is in the same directory, or else it will not work. Happy Programming!


