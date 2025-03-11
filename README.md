# Medical Imaging Coursework Solution 

This repository contains code, docs, report and graphs for the medical imaging coursework. 

## Getting Started

### Prerequisites

- Python 3.9.6
- `pip` (Python package installer)

### Setting Up a Virtual Environment

1. Open a terminal and navigate to root of this directory:



2. Create a virtual environment:
    try:
    ```sh
    python -m venv venv
    ```
    if above does not work: 
    ```sh
    python3 -m venv venv
    ```

3. Activate the virtual environment:

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

### Installing Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

### Documentation

The documentation for this project can be found in the `docs` directory. Open the `index.html` file in `docs/build/html` in the web browser to view the documentation.

### Running the Code

To run different files in the `code` directory, use the following commands:

- To run `CT` module:

    ```sh
    python code/run_script.py CT --data_directory data/Module3 --output_graph_directory graphs/

    ```
    if above does not work:
    ```sh
    python3 code/run_script.py CT --data_directory data/Module3 --output_graph_directory graphs/

    ```

- To run `MRI` module:

    ```sh
    python code/run_script.py MRI --data_directory data/Module2 --output_graph_directory graphs/

    ```
    if above does not work:
    ```sh
    python3 code/run_script.py MRI --data_directory data/Module2 --output_graph_directory graphs/

    ```

- To run `PET_CT` module:

    ```sh
    python code/run_script.py PET_CT --data_directory data/Module1 --output_graph_directory graphs/

    ```
    if above does not work:
    ```sh
    python3 code/run_script.py PET_CT --data_directory data/Module1 --output_graph_directory graphs/

    ```

