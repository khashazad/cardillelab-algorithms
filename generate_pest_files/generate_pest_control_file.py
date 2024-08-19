import json
import pandas as pd

def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def create_control_file(data, output_filename, observation_count):
    with open(output_filename, "w") as file:
        file.write("pcf\n")  
        ## Control Data ##

        file.write("* control data\n")

        # Line 1
        line1 = data["control_data"]["line1"]
        file.write(f"{line1['RSTFLE']['value']} {line1['PESTMODE']['value']}\n")

        # Line 2
        line2 = data["control_data"]["line2"]
        file.write(
            f"{len(data["parameter_data"])} {observation_count} {len(data["parameter_data"])} {line2['NPRIOR']['value']} 1\n"
        )

        # Line 3
        line3 = data["control_data"]["line3"]
        file.write(
            f"{line3['NTPLFLE']['value']} {line3['NINSFLE']['value']} {line3['PRECIS']['value']} {line3['DPOINT']['value']}\n"
        )

        # Line 4
        line4 = data["control_data"]["line4"]
        file.write(
            f"{line4['RLAMBDA1']['value']} {line4['RLAMFAC']['value']} {line4['PHIRATSUF']['value']} {line4['PHIREDLAM']['value']} {line4['NUMLAM']['value']}\n"
        )

        # Line 5
        line5 = data["control_data"]["line5"]
        file.write(
            f"{line5['RELPARMAX']['value']} {line5['FACPARMAX']['value']} {line5['ABSPARMAX']['value']}\n"
        )

        # Line 6
        line6 = data["control_data"]["line6"]
        file.write(
            f"{line6['PHIREDSWH']['value']}\n"
        )

        # Line 7
        line7 = data["control_data"]["line7"]
        file.write(
            f"{line7['NOPTMAX']['value']} {line7['PHIREDSTP']['value']} {line7['NPHISTP']['value']} {line7['NPHINORED']['value']} {line7['RELPARSTP']['value']} {line7['NRELPAR']['value']} {line7['PHISTOPTHRESH']['value']}\n"
        )

        # Line 8
        line8 = data["control_data"]["line8"]
        file.write(
            f"{line8['ICOV']['value']} {line8['ICOR']['value']} {line8['IEIG']['value']} {line8['IRES']['value']} {line8['JCOSAVE']['value']} {line8['JCOSAVEITN']['value']} {line8['VERBOSEREC']['value']} {line8['REISAVEITN']['value']} {line8['PARSAVEITN']['value']}\n"
        )

        # Singular Value Decompostion
        file.write("* singular value decomposition\n")
        svd = data["singular_value_decomposition"]

        file.write(f"{svd["line1"]['SVDMODE']['value']}\n")
        file.write(f"{svd["line2"]['MAXSING']['value']} {svd["line2"]['EIGTHRESH']['value']}\n")
        file.write(f"{svd["line3"]['EIGWRITE']['value']}\n")

        # Parameter Groups Section
        file.write("* parameter groups\n")
        for group in data["parameter_groups"]:
            file.write(f"{group['name']} {group['inctyp']} {group['derinc']} {group['derinclb']} {group['forcen']} {group['derincmul']} {group['splitthresh']}\n")
        
        # Parameter Data Section
        file.write("* parameter data\n")
        for param in data["parameter_data"]:
            file.write(f"{param['name']} {param['trans']} {param['inctyp']} {param['parval1']} {param['parlbnd']} {param['parubnd']} {param['pargp']} {param['scale']} {param['offset']}\n")

        # Observations groups
        file.write("* observation groups\n")
        file.write("obsgroup\n")

def read_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    return df

def parse_observations(df):
    observations = []
    previous_point_index = None
    observation_index = 1
    for index, row in df.iterrows():

        point_index = row['point']
        intercept = row['intercept']
        cos = row['cos']
        sin = row['sin']
        repetion_count = int(row['repetition'])

        if point_index != previous_point_index:
            observation_index = 1
            previous_point_index = point_index
        
        for i in range(repetion_count):
            observations.append((f"intercept_{int(point_index)}_{observation_index}", intercept))
            observations.append((f"cos_{int(point_index)}_{observation_index}", cos))
            observations.append((f"sin_{int(point_index)}_{observation_index}", sin))
            observation_index += 1

    return observations

def write_observations_to_control_file(observations, file_path):
    with open(file_path, 'a') as file:
        file.write("* observation data\n")
        for observation_name, observation_value in observations:
            file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")

def create_pest_instruction_file(observations, instructions_filename):
    with open(instructions_filename, "w") as file:
        file.write("pif *\n")
        file.write(f"l1\n")
        grouped_observation = [observations[i:i+3] for i in range(0, len(observations), 3)]
        for obs in grouped_observation:
            intercept = obs[0][0]
            cos = obs[1][0]
            sin = obs[2][0]

            file.write(f"l1 *,* !{intercept}! *,* !{cos}! *,* !{sin}!  *,*\n")

def append_model_and_io_sections(file_path):
    with open(file_path, 'a') as file:
        file.write("* model command line\n")
        file.write("model.bat\n")
        file.write("* model input/output\n")
        file.write("input.tpl  pest_input.csv\n")
        file.write("output.ins  pest_output.csv\n")

def create_template_file(template_filename):
    with open(template_filename, "w") as file:
        file.write("ptf #\n")
        file.write("#q1        #,0,0,0,#q5        #,0,0,0,#q9        #\n")
        file.write("#r         #\n")
        file.write("#p1        #,0,0,0,#p5        #,0,0,0,#p9        #\n")
        file.write("#x1        #,#x2        #,#x3        #\n")

if __name__ == "__main__":

    parameters = "./generate_pest_files/parameters.json"
    observations_filename = "./generate_pest_files/observations.csv"

    control_filename = "./generate_pest_files/output/eeek.pst"
    instructions_filename = "./generate_pest_files/output/output.ins"
    template_filename = "./generate_pest_files/output/input.tpl"

    data = read_json(parameters)
    
    df = read_csv(observations_filename)
    observations = parse_observations(df)
    
    create_control_file(data, control_filename, len([x for x in observations if x[1] != 0]))
    write_observations_to_control_file(observations, control_filename)
    create_pest_instruction_file(observations, instructions_filename)
    append_model_and_io_sections(control_filename)
    create_template_file(template_filename)
    print(f"Control files '{control_filename}' has been created.")

