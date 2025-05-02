import os
from logger import logger

def generate_filename_map(directory):
    """
    Generates a mapping of filenames in a directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        dict: A dictionary where the keys are filenames without extensions and 
              the values are their respective positions in the directory listing.
    """
    files = os.listdir(directory)
    filenames = [x[:x.find(".")] for x in files]
    filenames_map = {v: k for k, v in enumerate(filenames)}
    
    return filenames_map    

if __name__ == "__main__":
    mock_req_doc_raw_path = os.path.join("tests", "mocks", "mock_0_req_doc_raw_text")
    
    filename_map = generate_filename_map(mock_req_doc_raw_path)
    logger.debug(f"Filename_map: {filename_map}")