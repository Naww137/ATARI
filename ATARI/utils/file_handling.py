import os
import shutil


def clean_and_make_directory(directory):
    try:
        shutil.rmtree(directory)
    except:
        pass
    os.mkdir(directory)

def return_random_subdirectory(directory):
    return os.path.join(directory, str(os.urandom(15).hex()))
