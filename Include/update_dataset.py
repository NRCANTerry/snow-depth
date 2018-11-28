# import necessary packages
import numpy as np
import configparser
from scipy.stats import trimboth
import warnings

# function to update the dataset of registration matrices
# if the number of registrations is less than 50, the mean
# squared error will only be appended
# if the number is greater than 50 the mean and std dev will
# be calculated and kept updated
def createDataset(template_name, dataset, dataset_enabled):
    # disable FutureWarning from trimboth
    # caused by conflict of scipy 1.10 and numpy 1.15
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # determine if there are sufficient images to initialize the dataset
    if(not dataset_enabled and len(dataset[1]) >= 50):
        # if so get values
        dataset_values = np.array(dataset[1])

        # use 25% trimmed mean
        filtered_dataset_values = trimboth(dataset_values, 0.25)

        # filter out outliers using standard deviation
        #mean = np.mean(dataset_values)
        #std_dev = np.std(dataset_values)
        #filtered_dataset_values = dataset_values[abs(dataset_values - mean) < 2 * std_dev]

        # determine standard deviation and mean for filtered dataset
        filtered_mean = np.mean(filtered_dataset_values)
        filtered_std_dev = np.std(filtered_dataset_values)
        num_filtered_values = filtered_dataset_values.size

        # update dataset array
        dataset = [[filtered_mean, filtered_std_dev, num_filtered_values],[]]

        # output to user
        print("\n\nDataset Created:")
        print("Mean: %0.2f" % filtered_mean)
        print("Standard Deviation: %0.2f" % filtered_std_dev)
        print("Number of Values: %d" % num_filtered_values)

    # convert from numpy to list
    elif dataset_enabled: dataset = np.array(dataset).tolist()

    # write changes to config file
    config = configparser.ConfigParser()
    config.read('./AppData/preferences.cfg')
    config.set('Template Registration Dataset', template_name, \
        str(dataset).replace("array(", "").replace(")", ""))
    with open('./AppData/preferences.cfg', 'w') as configfile:
        config.write(configfile)

# function to update the dataset of tensor measurements
# if the number of tensor measurements is less than 50, the tensor measurement
# will only be appended
# if the number is greater than 50 the mean and std dev will
# be calculated and kept updated
def createDatasetTensor(template_name, dataset, dataset_enabled):
    # disable FutureWarning from trimboth
    # caused by conflict of scipy 1.10 and numpy 1.15
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # iterate through stakes
    for j, stake in enumerate(dataset):
        # determine if there are sufficient images to initialize the dataset
        if(not dataset_enabled[j] and len(stake[1]) >= 50):
            # if so get values
            dataset_values = np.array(stake[1])

            # use 25% trimmed mean
            filtered_dataset_values = trimboth(dataset_values, 0.25)

            # filter out outliers using standard deviation
            #mean = np.mean(dataset_values)
            #std_dev = np.std(dataset_values)
            #filtered_dataset_values = dataset_values[abs(dataset_values - mean) < 2 * std_dev]

            # determine standard deviation and mean for filtered dataset
            filtered_mean = np.mean(filtered_dataset_values)
            filtered_std_dev = np.std(filtered_dataset_values)
            num_filtered_values = filtered_dataset_values.size

            # update dataset array
            dataset[j] = [[filtered_mean, filtered_std_dev, num_filtered_values],[]]

            # output to user
            print("\n\nStake %d Dataset Created:" % j)
            print("Mean: %0.3f" % filtered_mean)
            print("Standard Deviation: %0.5f" % filtered_std_dev)
            print("Number of Values: %d" % num_filtered_values)

        # convert from numpy to list
        elif dataset_enabled[j]: dataset[j] = np.array(dataset[j]).tolist()

    # write changes to config file
    config = configparser.ConfigParser()
    config.read('./AppData/preferences.cfg')
    config.set('Tensor Dataset', template_name, \
        str(dataset).replace("array(", "").replace(")", ""))
    with open('./AppData/preferences.cfg', 'w') as configfile:
        config.write(configfile)
