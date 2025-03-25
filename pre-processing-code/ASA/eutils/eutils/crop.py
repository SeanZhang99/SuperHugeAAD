"""
This script is used to preprocess and crop EEG data for ASA project. 
It demonstrates the application of a high-pass filter to remove low-frequency noise and artifacts from the EEG signals. 
"""

# Import necessary libraries
import os

import os
import mne
import shutil

# Define the base path where the EEG data is stored
data_path = 'TEST_DATA'

# We initially collected EEG data from 25 subjects, but the data from 5 subjects who did not attend to the right side
# according to the questionnaire has been manually deleted.
for subject in range(1, 26):
    # Create a subject folder name with leading zeros for proper sorting
    subject_folder = f'S{subject:03d}'  

    # Define the file paths for the EEG, VHDR, and VMRK files of the first trial
    eeg_file = os.path.join(data_path, subject_folder, f'eegDataAll_Subject{subject_folder}_Trial1.eeg')
    vhdr_file = os.path.join(data_path, subject_folder, f'eegDataAll_Subject{subject_folder}_Trial1.vhdr')
    vmrk_file = os.path.join(data_path, subject_folder, f'eegDataAll_Subject{subject_folder}_Trial1.vmrk')

    # Read the EEG data using the VHDR file, preloading the data for efficiency
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # Read annotations from the VMRK file
    annots = mne.read_annotations(vmrk_file)
    # Generate events from the annotations
    events, event_id = mne.events_from_annotations(raw)

    # Print the events and event IDs for the current subject
    print(f'Events for {subject_folder}:')
    print(events, event_id)
    
    # Apply a high-pass filter to the raw EEG data to remove low-frequency noise
    raw.filter(l_freq=1., h_freq=None)
    
    # Initialize a counter for the trials
    num = 0    
    # Iterate over the events to crop the data into segments
    for i in range(1, 85):
        # Skip every third event (i % 3 == 2), which might be a way to exclude certain types of events
        if i % 3 == 2:
            num += 1  
            # Calculate the start and end times for the current segment
            startTime = raw.times[events[i][0]]
            endTime = raw.times[events[i + 1][0]]
            
            # Print the duration of the current segment
            print(endTime - startTime)
            
            # Crop the raw data to the current segment
            raw_crop = raw.copy().crop(tmin=startTime, tmax=endTime)
            
            # Determine the save directory based on the trial number. In this project (ASA dataset), only E1 data is used.
            if num <= 20:
                save_dir = os.path.join(data_path, subject_folder, 'E1')
                file_name = f'{subject_folder}_E1_Trial{num}_raw.fif'
            else:
                save_dir = os.path.join(data_path, subject_folder, 'E2')
                file_name = f'{subject_folder}_E2_Trial{num - 20}_raw.fif'
            
            # Ensure the save directory exists, and create it if it doesn't
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # If it's the first or 21st trial and the directory already exists, remove it and recreate it
            if (num == 1 or num == 21) and os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
                
            # Construct the full file path for saving the cropped data
            file_path = os.path.join(save_dir, file_name)
            # Save the cropped data to a .fif file, overwriting any existing file
            raw_crop.save(file_path, overwrite=True)