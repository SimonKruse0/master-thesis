import numpy as np
import os
import pandas as pd

path = "/home/simon/Documents/MasterThesis/master-thesis/exdata"
solvers = []
for solver in os.listdir(path):
    solvers.append(solver)
    #indexes.append(f"{solver}_f_evals")

dims = [f"DIM{i}" for i in [2, 3, 5, 10, 20, 40]]
multiindexes = [(x,y,z) for x in solvers for y in dims for z in ["f", "f_evals"]]
index = pd.MultiIndex.from_tuples(multiindexes, names=["first", "second", "third"])
best_solutions = pd.DataFrame(index=index,columns=[f"f{i+1}" for i in range(24)])

def get_best_solutions(solver):
    search_path = os.path.join(path, solver)
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.endswith(".dat"):
                problem = root.split("_")[-1]
                dim = file.split("_")[-2]
                data_path = os.path.join(root, file)
                datContent = [i.strip().split("|") for i in open(data_path).readlines()]
                if datContent == []:
                    continue
                F = [float(l[0].split(" ")[2]) for l in datContent[1:]] #best noise-free fitness - Fopt (7.948000000000e+01)
                f = [float(l[0].split(" ")[3]) for l in datContent[1:]] # measured fitness 
                f_evals = [int(l[0].split(" ")[0]) for l in datContent[1:]]
                best_solutions[problem][solver, dim, "f"] = F[-1]
                best_solutions[problem][solver, dim, "f_evals"] = f_evals[-1]
    return best_solutions



def get_best_solutions_for_all_solvers():


    for solver in solvers:
        best_solutions = get_best_solutions(solver)
    return best_solutions

if __name__ == "__main__":
    best_solutions = get_best_solutions_for_all_solvers()
    print(best_solutions)
