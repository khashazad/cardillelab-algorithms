from utils.filesystem import read_json, read_file

def parse_initial_parameters(pest_file_path):
    params = {}
    lines = read_file(pest_file_path)

    for line in lines[23:27]:
        parts = line.split()
        param_name = parts[0]
        param_value = parts[3]
        params[param_name] = param_value

    return params["q1"], params["q5"], params["q9"], params["r"]

def parse_initial_and_final_objective_function(pest_file_path):
    with open(pest_file_path, "r") as file:
        lines = file.readlines()
        objective_functions = [line.split("=")[1].strip() for line in lines if "Sum of squared weighted residuals" in line]

        return float(objective_functions[0]), float(objective_functions[-1])

def parse_optimized_parameters(json_file_path):
    data = read_json(json_file_path)

    q1 = data["process noise (Q)"][0][0]
    q5 = data["process noise (Q)"][1][1]
    q9 = data["process noise (Q)"][2][2]
    r = data["measurement noise (R)"][0][0]

    return q1, q5, q9, r