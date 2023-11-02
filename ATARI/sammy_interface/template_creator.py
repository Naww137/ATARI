import numpy as np


def make_input_template(filepath, 
                model,
                experiment,
                rto,#: Union[SammyInputData, SammyInputDataYW],
                alphanumeric = []):
    
    alphanumeric_base = ["TWENTY", 
                        "EV", 
                        "GENERATE PLOT FILE AUTOMATICALLY",
                        "%%%alphanumeric%%%"]
    
    # if rto.options["bayes"]:
    #     bayes_cmd = "SOLVE BAYES EQUATIONS"
    # else:
    #     bayes_cmd = "DO NOT SOLVE BAYES EQUATIONS"
    # alphanumeric = [model.formalism, bayes_cmd] + experiment.inputs['alphanumeric'] + alphanumeric_base + alphanumeric
    alphanumeric = alphanumeric + alphanumeric_base + experiment.inputs['alphanumeric']

    if np.any([each.lower().startswith("broadening is not wa") for each in alphanumeric]):
        broadening = False
    else:
        broadening = True


    with open(filepath,'w') as f:
            # cards 1,2,3
            f.write(f"Title\n")
            f.write('%%%card2%%%\n')
            # f.write(f"{model.isotope: <9} {model.amu: <9} {float(min(experiment.energy_range)): <9} {float(max(experiment.energy_range)): <9}      {rto.options['iterations']: <5} \n")

            for cmd in alphanumeric:
                f.write(f'{cmd}\n')
            f.write('\n')

            # cards 5, 6
            # if broadening:
            #     f.write(f'  {float(experiment.parameters["temp"][0]):<8}  {float(experiment.parameters["FP"][0]):<8}  {float(experiment.parameters["FP"][1]):<8}        \n')
            f.write('%%%card5/6%%%\n')

            # card7
            # f.write(f'  {float(model.ac):<8}  {float(experiment.parameters["n"][0]):<8}                       0.00000          \n')
            f.write('%%%card7%%%\n')

            # card 8 
            # f.write(f'{experiment.reaction}')
            f.write('%%%card8%%%')

            # spin groups
            f.write(model.spin_groups)

            # ResFunc 
            if experiment.inputs['ResFunc'] is not None:
                f.write(f"\n{experiment.inputs['ResFunc']}\n")
                for resfuncline in experiment.get_resolution_function_lines():
                    f.write(f"{resfuncline}\n")
