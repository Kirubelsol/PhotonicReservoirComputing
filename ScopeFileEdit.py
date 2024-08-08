import numpy as np
import pandas as pd

#Reformat the csv file collected by the Oscilloscope for training and testing. 

def load_and_average(file_path, start_idx, end_idx, m, output_file):
    # Load the file
    data = pd.read_csv(file_path, header=None)
    print(start_idx)
    # Extract the 2nd and 3rd columns
    col2 = data.iloc[start_idx:end_idx, 1].to_numpy()
    col3 = data.iloc[start_idx:end_idx, 2].to_numpy()

   
    original_length = len(col2)
    indices = np.linspace(0, original_length, m + 1).astype(int)
    
    # Initialize the new arrays for the averaged values
    averaged_col2 = []
    averaged_col3 = []

    for i in range(m):
        # Determine the segment range
        segment_start = indices[i]
        segment_end = indices[i + 1]
        # Determine the middle half of the segment
        middle_start = segment_start + (segment_end - segment_start) // 4
        middle_end = segment_end - (segment_end - segment_start) // 4

        # Calculate the average for the current segment's middle half
        segment_avg2 = np.mean(col2[middle_start:middle_end])
        segment_avg3 = np.mean(col3[middle_start:middle_end])

        # Append the averages to the new arrays
        averaged_col2.append(segment_avg2)
        averaged_col3.append(segment_avg3)

    averaged_data = pd.DataFrame(np.column_stack((averaged_col2, averaged_col3)))
    averaged_data.to_csv(output_file, index=False, header=False)

    return (averaged_col2)

# Function to rearrange the data
def rearrange_data(first_column, output_file, elements_per_row=50):

    num_rows = len(first_column) // elements_per_row
    reshaped_data = np.reshape(first_column[:num_rows * elements_per_row], (num_rows, elements_per_row))

    reshaped_df = pd.DataFrame(reshaped_data)
    reshaped_df.to_csv(output_file, index=False, header=False)


# Example usage
variable = 30
avg = 2
file_path = 'tr7.csv'  # Replace with your file path
start_idx = 70068
end_idx = 124232
m = 3000
output_file = 'tr7_avg2.csv'  # Replace with your desired output file path
output_file_rearrange = 'tr7_re2.csv'
# Call the function and save the results
col2 = load_and_average(file_path, start_idx, end_idx, m, output_file) 
rearrange_data(col2, output_file_rearrange, elements_per_row=50)


