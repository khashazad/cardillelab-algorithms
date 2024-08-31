
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
            f"{len(data["parameter_data"])} {observation_count} {len(data["parameter_data"])} {line2['NPRIOR']['value']} {line2['NOBSGP']['value']}\n"
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

def append_observations_to_control_file(observations, file_path, observation_flags):
    with open(file_path, 'a') as file:
        file.write("* observation data\n")
        for observation_name, observation_value in observations:
            if observation_name.startswith("intercept") and observation_flags["intercept"]:
                file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")
            elif observation_name.startswith("cos") and observation_flags["cos"]:
                file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")
            elif observation_name.startswith("sin") and observation_flags["sin"]:
                file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")
            elif observation_name.startswith("estimate") and observation_flags["estimate"]:
                file.write(f"{observation_name.ljust(15)} {str(observation_value).ljust(15)} 1.0 obsgroup\n")

def append_model_and_io_sections_to_control_file(control_filename):
    with open(control_filename, 'a') as file:
        file.write("* model command line\n")
        file.write("model.bat\n")
        file.write("* model input/output\n")
        file.write("input.tpl  pest_input.csv\n")
        file.write("output.ins  pest_output.csv\n")

def create_instructions_file(observations, instructions_filename, observation_flags):
    with open(instructions_filename, "w") as file:
        file.write("pif *\n")
        file.write(f"l1\n")
        grouped_observation = [observations[i:i+4] for i in range(0, len(observations), 4)]
        for obs in grouped_observation:
            intercept = obs[0][0]
            cos = obs[1][0]
            sin = obs[2][0]
            estimate = obs[3][0]

            intercept_value = f"!{intercept}!" if observation_flags["intercept"] else ""
            cos_value = f"!{cos}!" if observation_flags["cos"] else ""
            sin_value = f"!{sin}!" if observation_flags["sin"] else ""
            estimate_value = f"!{estimate}!" if observation_flags["estimate"] else ""
            file.write(f"l1 *,* {intercept_value} *,* {cos_value} *,* {sin_value} *,* {estimate_value}\n")


def create_template_file(template_filename):
    with open(template_filename, "w") as file:
        file.write("ptf #\n")
        file.write("#q1        #,0,0,0,#q5        #,0,0,0,#q9        #\n")
        file.write("#r         #\n")
        file.write("#p1        #,0,0,0,#p5        #,0,0,0,#p9        #\n")

def create_model_file(file_path):
    with open(file_path, "w") as file:
        file.write(r"python C:\Users\kazad\OneDrive\Documents\GitHub\eeek\pest_eeek.py --input=pest_input.csv --output=pest_output.csv --points=points.csv --num_sinusoid_pairs=1 --include_intercept --store_measurement --collection=L8_L9_2022_2023 --store_estimate --store_date")
