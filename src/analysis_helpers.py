import os
import json
import matplotlib.pyplot as plt

BASE_DIRECTORY = '/home/simon/Documents/MasterThesis/master-thesis'

def get_data2(target_name, use_exact_name=False):
    data_list = []
    name_list = []
    file_path_list = set()
    problem_name = None
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        # Opening JSON file
        #print(dirpath, filenames)
        for filename in filenames:
            if "noise" in filename:
                filename2 = filename
                filename = filename.replace("noise_", "noise-")

            if target_name not in filename:
                continue
            if (target_name != "_".join(filename.split("_")[1:4])) and use_exact_name:
                continue
            try:
                file_path = os.path.join(dirpath, filename)
                with open(file_path) as json_file:
                    data_list.append(json.load(json_file))
                    name_list.append(filename.split("_")[0])
                    file_path_list.add(dirpath.split("/")[-1])
                    tmp =  " ".join(filename.split("_")[1:4])
                    if problem_name is None:
                        problem_name = tmp
                    else:
                        if tmp != problem_name:
                            print("target", target_name)
                            print("More problems in same target..", problem_name,"and", tmp)
                            return 0
            except:
                try:
                    file_path = os.path.join(dirpath, filename2)
                    with open(file_path) as json_file:
                        data_list.append(json.load(json_file))
                        name_list.append(filename.split("_")[0])
                        file_path_list.add(dirpath.split("/")[-1])
                        tmp =  " ".join(filename.split("_")[1:4])
                        if problem_name is None:
                            problem_name = tmp
                        else:
                            if tmp != problem_name:
                                print("target", target_name)
                                print("More problems in same target..", problem_name,"and", tmp)
                                return 0
                except:
                    print(filename, "could not be read")

    file_path_list2 = [f"{x[2:4]}/{x[:2]} {x[-4:-2]}:{x[-2:]}" for x in file_path_list]
    return data_list, name_list, problem_name, file_path_list, file_path_list2

def get_names():
    names = set()
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        
        # Opening JSON file
        #print(dirpath, filenames)
        for filename in filenames:
            if "noise" in filename:
                filename = filename.replace("noise_", "noise-")
            names.add("_".join(filename.split("_")[1:4]))

    return sorted(names)