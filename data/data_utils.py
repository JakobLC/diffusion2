import os
import json
import zipfile
import tarfile
import gzip
import jsonlines
import numpy as np

def unpack_files(filename):
    base_dir = os.path.dirname(filename)
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".zip":
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        return f"Unpacked {filename} successfully."

    elif file_extension == ".tar":
        with tarfile.open(filename, 'r') as tar_ref:
            tar_ref.extractall(base_dir)
        return f"Unpacked {filename} successfully."

    elif file_extension == ".gz":
        output_filename = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(base_dir, output_filename)
        with gzip.open(filename, 'rb') as gz_ref:
            with open(output_path, 'wb') as output_file:
                output_file.write(gz_ref.read())
        return f"Unpacked {filename} successfully."

    else:
        return f"Unsupported file format: {file_extension}. Cannot unpack."


def save_dict_list_to_json(data_list, file_path, append=False):
    assert len(file_path)>=5, "File path must end with .json or .jsonl"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if not isinstance(data_list,list):
        data_list = [data_list]
    if file_path[-5:] == ".json":
        if append:
            try:
                existing_data = load_json_to_dict_list(file_path)
                combined_data = existing_data + data_list
            except FileNotFoundError:
                combined_data = data_list
        else:
            combined_data = data_list
        
        with open(file_path, 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)
    elif file_path[-6:] == ".jsonl":
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)
    else:
        raise ValueError("File path must end with .json or .jsonl")
        
    

def load_json_to_dict_list(file_path):
    assert len(file_path)>=5, "File path must end with .json"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if file_path[-5:] == ".json":
        with open(file_path, 'r') as json_file:
            data_list = json.load(json_file)
    elif file_path[-6:] == ".jsonl":
        data_list = []
        with jsonlines.open(file_path) as reader:
            for line in reader:
                data_list.append(line)
    return data_list

def rle_to_mask(rle):
    from pycocotools import mask as mask_utils
    h, w = rle[0]["segmentation"]["size"]
    mask = np.zeros((h,w),dtype=np.uint8)
    i = 1
    for d in rle:
        d = mask_utils.decode(d["segmentation"])
        mask[d==1] = i
        i += 1
    return mask