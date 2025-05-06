import os
from lisa.logger import logger

def generate_filename_map(directory):
    """
    Generates two mappings of filenames in a directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        tuple[dict, dict]: 
            - filenames_map: keys = names without extensions, values = index
            - map_filenames: keys = index, values = names without extensions
    """
    filenames_map = {}
    map_filenames = {}
    
    files = sorted(os.listdir(directory))
    for i, file in enumerate(files):
        filename = os.path.splitext(file)[0]
        filenames_map[filename] = str(i)
        #map_filenames[str(i)] = filename
        
    return filenames_map 

if __name__ == "__main__":
    mock_req_doc_raw_path = os.path.join("tests", "mocks", "mock_0_req_doc_raw_text")
    
    filename_map = generate_filename_map(mock_req_doc_raw_path)
    logger.debug(f"Filename_map: {filename_map}")