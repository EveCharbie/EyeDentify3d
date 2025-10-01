import os
from pathlib import Path
import pandas as pd
import pandas.testing as pdt


def test_complete_example():

    # Check tha the file does not exist
    current_path_file = Path(__file__).parent
    result_path = f"{current_path_file}/../examples/results/HTC_Vive_Pro/full_trial_results.csv"
    if os.path.exists(result_path):
        os.remove(result_path)

    # Run the complete example, which will generate a data frame of all results and save it to a CSV file
    from examples.complete_example import perform_all_files

    perform_all_files()

    # Load the results and compare with a reference file
    test_results = pd.read_csv(result_path)
    reference_results = pd.read_csv(result_path.replace(".csv", "_reference.csv"))

    for i_line in range(len(test_results)):
        participant_id = test_results.iloc[i_line]["participant_id"]
        trial_id = test_results.iloc[i_line]["trial_id"]

        # Get the test row
        test_row = test_results.iloc[[i_line]]

        # Find matching row in reference by participant_id and trial_id
        reference_row = reference_results[
            (reference_results["participant_id"] == participant_id) & (reference_results["trial_id"] == trial_id)
        ]

        # Reset index for proper comparison
        pdt.assert_frame_equal(test_row.reset_index(drop=True), reference_row.reset_index(drop=True), check_dtype=False)
