import os
import shutil
import json

def delete_existing_directory_and_create_new(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def read_file(filename):
    with open(filename, "r") as file:
        data = file.readlines()
    return data