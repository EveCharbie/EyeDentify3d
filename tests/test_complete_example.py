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
    pdt.assert_frame_equal(test_results, reference_results, check_dtype=False)


