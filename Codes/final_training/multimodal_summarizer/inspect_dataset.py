import pandas as pd
import pickle
import sys
import os
import io # Import the io module

# --- CONFIGURATION ---
# IMPORTANT: Update this path to point to ONE of your .p files
PICKLE_FILE_PATH = r"C:\Users\admin\Downloads\articles\articles\processed_bbc_1.p" # Example path
OUTPUT_FILE_NAME = "inspection_results.txt"
# ---------------------

def inspect_pickle_file(file_path, output_file):
    """
    Loads a pickle file and intelligently inspects its content,
    saving all output to a specified text file.
    """
    # Redirect stdout to a string buffer
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        print(f"--- Inspecting Pickle File ---")
        print(f"File: {file_path}\n")

        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            print("Please update the PICKLE_FILE_PATH variable in this script.")
            return

        try:
            # Load the pickle file using the standard pickle library
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Successfully loaded pickle file.")
            data_type = type(data)
            print(f"Object type: {data_type}")

            # --- Handle different data types ---
            
            if isinstance(data, pd.DataFrame):
                # If it IS a DataFrame
                print("\n--- DataFrame Info ---")
                # data.info() prints directly, so it's captured
                data.info() 
                
                print("\n--- DataFrame Columns ---")
                print(data.columns.tolist())
                
                print("\n--- First Row (Example) ---")
                if not data.empty:
                    pd.set_option('display.max_colwidth', 300)
                    # .to_string() returns a string, so we explicitly print it
                    print(data.head(1).to_string()) 
                else:
                    print("DataFrame is empty.")
            
            elif isinstance(data, dict):
                # If it's a Dictionary
                print(f"Data is a dictionary with {len(data.keys())} keys.")
                print("\n--- Dictionary Keys (First 50) ---")
                print(list(data.keys())[:50])
                
                # Try to print the first item
                if data.keys():
                    first_key = list(data.keys())[0]
                    print("\n--- First Item (Example) ---")
                    print(f"Key: {first_key}")
                    
                    # Pretty print the value if it's a dict or list
                    value = data[first_key]
                    if isinstance(value, (dict, list)):
                        import json
                        print(f"Value (formatted):\n{json.dumps(value, indent=2)}")
                    else:
                        print(f"Value: {value}")
                else:
                    print("Dictionary is empty.")
            
            elif isinstance(data, (list, tuple)):
                # If it's a List or Tuple
                print(f"Data is a {data_type} with {len(data)} items.")
                if data:
                    print("\n--- First Item (Example) ---")
                    print(data[0])
                else:
                    print("List/Tuple is empty.")
            
            else:
                # Other types
                print("\n--- Data Sample (first 500 chars) ---")
                print(str(data)[:500] + "...")

        except (pickle.UnpicklingError, AttributeError) as e:
            print(f"Error: Failed to unpickle/read file. The file may be corrupt or in an unexpected format. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during inspection: {e}")

    finally:
        # Restore stdout
        sys.stdout = old_stdout 
        
        # Get the captured output
        output_content = redirected_output.getvalue()
        
        # Save the captured output to a file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"--- Inspection Complete ---")
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")
            print("\n--- Inspection Output (Fallback) ---")
            print(output_content)


if __name__ == "__main__":
    inspect_pickle_file(PICKLE_FILE_PATH, OUTPUT_FILE_NAME)