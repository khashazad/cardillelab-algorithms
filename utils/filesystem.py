import os
import shutil
import json


def delete_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def read_file(filename):
    with open(filename, "r") as file:
        data = file.readlines()
    return data


def write_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
