import os
import argparse
import importlib.util


def run_scripts_in_folder(folder_path, input_path, output_path):
    """Run all Python scripts in the specified folder with input and output paths.
    Parameters:
    - folder_path (str): The path to the folder containing the scripts.
    - input_path (str): The path to the input data.
    - output_path (str): The path to the output data.
    
    Returns:
    - None
    """
    print(f"Running scripts in folder: {folder_path}")
    print(f"Using input path: {input_path}")
    print(f"Using output path: {output_path}")
    
    # Get all Python files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.py')]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Run each script
    for script in files:
        script_path = os.path.join(folder_path, script)
        script_name = os.path.splitext(script)[0]
        print(f"\nRunning: {script}")

        spec = importlib.util.spec_from_file_location(script_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(input_path, output_path)


    


def main():
    parser = argparse.ArgumentParser(description="Run Python scripts from a specific folder")
    parser.add_argument("folder", choices=["MRI", "CT", "PET_CT"], 
                        help="Folder containing scripts to run")
    parser.add_argument("--data_directory", required=True, help="Path to input data")
    parser.add_argument("--output_graph_directory", required=True, help="Path to output graph")
    
    args = parser.parse_args()
    
    # Determine the folder path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, args.folder)
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_path} not found or is not a directory")
        return
    
    # Run all scripts in the folder with input and output paths
    run_scripts_in_folder(folder_path, args.data_directory, args.output_graph_directory)

if __name__ == "__main__":
    main()