
import os
import sys
import numpy as np
from pathlib import Path
import importlib.util
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('./plot_style.txt')
from matplotlib.ticker import MaxNLocator

from Discrete_Chaos.DiscreteChaos import DiscreteChaosSuite as DCS

def select_map():
    with st.form("Map selection"):
        st.markdown("# Map Selection")
        if not st.session_state.get('num_equations'):
            st.session_state['num_equations'] = 1
        if not st.session_state.get('num_parameters'):
            st.session_state['num_parameters'] = 1
        num_equations = st.number_input("System Dimension", min_value = 1, step = 1, value=st.session_state.get('num_equations'))
        num_parameters = st.number_input("Number of parameters", min_value = 1, step = 1, value=st.session_state.get('num_parameters'))

        select_map = st.form_submit_button("Select map")
        
    if select_map:
        st.session_state["equations_selected"] = False
        st.session_state["select_map"] = True
        st.session_state["names_selected"] = None
        st.session_state["iteration"] = None
        st.session_state["variable_names"] = None
        st.session_state["parameter_names"] = None
        # st.session_state["iteration_text"] = iteration_function
        st.session_state['num_equations'] = num_equations
        st.session_state['num_parameters'] = num_parameters
        base_figure_dict = dict(zip(range(num_equations), num_equations * [None]))
        st.session_state["bifurcation_diagrams"] = base_figure_dict.copy()
        st.session_state["lyapunov_exponent"] = base_figure_dict.copy()
        st.session_state["trajectory_diagrams"] = base_figure_dict.copy()
        st.session_state["cobweb_diagrams"] = base_figure_dict.copy()

        st.session_state["params_submitted"] = False
        st.session_state["trajectory_params_submitted"] = False

    if st.session_state.get("select_map"):
        num_eqs = st.session_state['num_equations']
        num_pars = st.session_state['num_parameters']
        with st.form("Variable and parameter names"):
            var_names = num_eqs * [""]
            var_name_cols = st.columns(num_eqs)
            for n, col in enumerate(var_name_cols):
                var_names[n] = col.text_input(f"Variable {n + 1} name", max_chars=1)
            
            par_names = num_pars * [""]
            par_name_cols = st.columns(num_pars)
            for n, col in enumerate(par_name_cols):
                par_names[n] = col.text_input(f"Parameter {n + 1} name", max_chars=1)

            names_button_cols = st.columns(2)
            select_names = names_button_cols[0].form_submit_button("Select names")
            cancel = names_button_cols[1].form_submit_button("Cancel")
    
        if select_names:
            st.session_state["names_selected"] = True
            if "" in var_names:
                st.error("Variable Name not given")
                st.session_state["names_selected"] = False
            elif len(list(set(var_names))) < len(var_names):
                st.error("Same name was given for two variables")
                st.session_state["names_selected"] = False
            else:
                var_names = [v.strip() for v in var_names]
                st.session_state["variable_names"] = var_names

            if "" in par_names:
                st.error("Parameter Name not given")
                st.session_state["names_selected"] = False
            elif len(list(set(par_names))) < len(par_names):
                st.error("Same name was given for two variables")
                st.session_state["names_selected"] = False
            else:
                par_names = [v.strip() for v in par_names]
                parameter_names_ok = True
                for par in par_names:
                    if par in st.session_state["variable_names"]:
                        st.error("Identical name found for variable and parameter")
                        st.session_state["names_selected"] = False
                        parameter_names_ok = False
                if parameter_names_ok:
                    st.session_state["parameter_names"] = par_names

        if cancel:
            st.session_state["select_map"] = False
            st.session_state["names_selected"] = False
            st.experimental_rerun()

    if st.session_state.get("names_selected"):
        with st.form("Insert Equations"):
            variable_names = st.session_state["variable_names"]
            num_eqs = len(variable_names)
            equations = num_eqs * [""]
            for n in range(num_eqs):
                equations[n] = st.text_input(f"Equation {n+1}")

            eqn_button_cols = st.columns(2)
            select_equations = eqn_button_cols[0].form_submit_button("Select equations")
            cancel = eqn_button_cols[1].form_submit_button("Cancel")
            
        if select_equations:
            if "" in equations:
                st.error("State equation is not given")
                st.session_state["equations_selected"] = False
            else:
                eqns = [eq.strip() for eq in equations]
                st.session_state["equations_selected"] = True
                st.session_state["equations_text"] = eqns

        if cancel:
            st.session_state["names_selected"] = False
            st.session_state["equations_selected"] = False
            st.experimental_rerun()

		
		with st.expander("Example equations"):
			example_cols = st.columns(2)
			example_cols[0].write("Logistic Map")
			example_cols[1].write("r * x * (1 -x)")
			
			example_cols[0].write("Sine Map")
			example_cols[0].write("m * sin(pi * x)")
			
			example_cols[0].write("")
			example_cols[0].write("")
			
			example_cols[0].write("Arnold Map")
			example_cols[0].write("mod(2 * x + y, 1)")
			example_cols[0].write("mod(x + y, 1)")
			

    if st.session_state.get("equations_selected"):

        curr_dir = os.getcwd()
        file_name = curr_dir + "/temp.py"


        with open(file_name, "w") as file:
            tab = 4 * " "
            var_names = st.session_state["variable_names"]
            par_names = st.session_state["parameter_names"]


            file.write("from numpy import *\n")
            file.write("def iteration(xx, rr): \n")
            if len(var_names) > 1:
                file.write(tab + f'{",".join(var_names)} = xx\n')
            else:
                file.write(tab + f"{var_names[0]}, = xx\n")
            
            if len(par_names) > 1:
                file.write(tab + f'{",".join(par_names)} = rr\n')
            else:
                file.write(tab + f"{par_names[0]}, = rr\n")

            eqn_texts = st.session_state["equations_text"]
            for n, eq in enumerate(eqn_texts):
                file.write(tab + f"{var_names[n]} = {eq}\n")

            if len(var_names) > 1:
                file.write(tab + f'return {",".join(var_names)}')
            else:
                file.write(tab + f'return {var_names[0]},')
            
            


        path = Path(file_name)

        spec = importlib.util.spec_from_file_location(path.stem, path)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = foo
        spec.loader.exec_module(foo)

        iteration = foo.iteration

        st.session_state["iteration"] = iteration
        os.remove(file_name)

        
        iteration = st.session_state["iteration"]
        chaotic_map = DCS(iteration)
        st.session_state["chaotic_map"] = chaotic_map
		
		st.success("Map selected")


###########################################################################################################




def trajectories():
    if st.session_state.get("chaotic_map") is None:
        st.error("Please select a map first")
        return 
    chaotic_map = st.session_state["chaotic_map"]

    with st.form("Trajectory Initial Conditions"):
        st.markdown("# Initial conditions and parameters")
        initial_conditions_string = st.text_input("Initial Condition")
        parameters_string = st.text_input("Parameter Values")
        num_points = st.number_input("Number of points", min_value = 20, step = 1, value=50)
        submit_traj_params = st.form_submit_button("Submit Trajectory Parameters")

    if submit_traj_params:
        st.session_state["single_le"] = None
        st.session_state["show_trajectory"] = False
        num_equations = st.session_state['num_equations']
        base_figure_dict = dict(zip(range(num_equations), num_equations * [None]))
        st.session_state["trajectory_diagrams"] = base_figure_dict.copy()
        st.session_state["cobweb_diagrams"] = base_figure_dict.copy()
        if not initial_conditions_string:
            st.error("Please enter initial conditions")
            return
        if not parameters_string:
            st.error("Please enter parameter values")
            return
        

        num_equations = st.session_state["num_equations"]
        st.session_state["trajectory_params_submitted"] = True
        init_list = initial_conditions_string.split(",")
        init_list = list(map(lambda x: x.strip(), init_list))
        init_cond = tuple([float(x) for x in init_list if x])

        params_list = parameters_string.split(",")
        params_list = list(map(lambda x: x.strip(), params_list))
        params = tuple([float(x) for x in params_list if x])
        st.session_state["trajectory_initial_condition"] = init_cond
        st.session_state["trajectory_parameters"] = params   

    if st.session_state["trajectory_params_submitted"]:
        with st.form("bifurcation diagram"):
            num_equations = st.session_state["num_equations"]
            chosen_variable = st.selectbox("Which variable", range(1, 1 + num_equations))
            #fixme This should be a list from multiselection

            show_timeseries = st.checkbox("Show timeseries")
            traj_button = st.form_submit_button("Selected Variable")
            
        if traj_button:
            st.session_state["show_trajectory"] = True
            st.session_state["show_timeseries"] = show_timeseries

    
        if st.session_state.get("show_trajectory"):

            chaotic_map = st.session_state["chaotic_map"]
            init_cond = st.session_state["trajectory_initial_condition"]
            params = st.session_state["trajectory_parameters"]

            if st.session_state.get("single_le") is None:
                st.session_state["single_le"] = chaotic_map.lyapunov_exponents_approximate_qr(init_cond, params)

            st.write(f"Lyapunov Exponent {st.session_state.get('single_le')[chosen_variable-1]}")

            if st.session_state["trajectory_diagrams"].get(chosen_variable) is not None and not st.session_state["show_timeseries"]:
                print("I am here")
                plt.figure(st.session_state["trajectory_diagrams"][chosen_variable])
                st.pyplot(st.session_state["trajectory_diagrams"][chosen_variable])
            else:
                chaotic_map = st.session_state["chaotic_map"]
                init_cond = st.session_state["trajectory_initial_condition"]
                params = st.session_state["trajectory_parameters"]
                fig = plt.figure()
                d = chaotic_map.sequence_gen_finite(init_cond, params, num_points+1)
                for var in [chosen_variable]:
                    st.markdown(f"Trajectory of variable {var}")
                    fig = plt.figure()
                    points = [point[var-1] for point in d]
                    n = len(points)
                    plt.plot(range(n), points)
                    plt.autoscale(enable=True, axis='x', tight=True)
                    plt.ylabel(st.session_state.get('variable_names')[var-1], rotation=0)
                    plt.xlabel("t")
                    plt.tight_layout()
                    st.session_state["trajectory_diagrams"][chosen_variable] = fig
                    st.pyplot(fig)
            
                if st.session_state["show_timeseries"]:
                    st.write(", ".join(list(map(str, points))))

###################################################
def dynamical_analysis():
    if st.session_state.get("chaotic_map") is None:
        st.error("Please select a map first")
        return
    with st.form("Initial Conditions"):
        st.markdown("# Initial conditions and parameters")
        initial_conditions_string = st.text_input("Initial Condition")
        num_equations = st.session_state.get("num_equations")
        num_parameters = st.session_state.get("num_parameters")

        which_parameter = st.selectbox("Which parameter", tuple(range(1, 1 + num_parameters)))
        which_parameter = int(which_parameter)
        param_select = st.form_submit_button("Select Parameter")
    
    if param_select:
        st.session_state["parameter_selected"] = True
        if initial_conditions_string:
            init_list = initial_conditions_string.split(",")
            init_list = list(map(lambda x: x.strip(), init_list))
            init_cond = tuple([float(x) for x in init_list if x])
            st.session_state["initial_condition"] = init_cond
        else:
            st.error("Please enter initial conditions")
            return
        
        st.session_state["which_parameter"] = which_parameter
        st.session_state["parameters"] = ()

        base_figure_dict = dict(zip(range(num_equations), num_equations * [None]))
        st.session_state["bifurcation_diagrams"] = base_figure_dict.copy()
        st.session_state["lyapunov_exponent"] = base_figure_dict.copy()

    if st.session_state.get("parameter_selected"):
        with st.form("Select Parameter Values"):
            parameters = {}
            for i in range(1, 1 + num_parameters):
                if not i == which_parameter:
                    parameters[i-1] = st.number_input(f"Value for parameter {i}", format = "%.5f")
                elif i == which_parameter:
                    pam_cols = st.columns(3)
                    parameter_start = pam_cols[0].number_input(f"Parameter {i} range start", value = 1.0, min_value = 1e-16, max_value = 1e6, format = "%.5f")
                    parameter_end = pam_cols[1].number_input(f"Parameter {i} range end", value = 2.0, min_value = 1e-15, max_value = 1e06, format = "%.5f")
                    parameter_step = pam_cols[2].number_input(f"Parameter {i} step", value = 1e-01, min_value = 1e-15, max_value = 1e03, format = "%.10f")
                    parameters[i-1] = np.arange(parameter_start, parameter_end, parameter_step)
            
            params = [parameters.get(i) for i in range(num_parameters)]
            params = tuple(params)
            submit_param = st.form_submit_button("Select")
            if submit_param:
                print("I am here")

                st.session_state["parameters_chosen"] = True
                st.session_state["parameters"] = params
                    
                st.success("Parameters Chosen Successfully")
                
                base_figure_dict = dict(zip(range(num_equations), num_equations * [None]))
                st.session_state["bifurcation_diagrams"] = base_figure_dict.copy()
                st.session_state["lyapunov_exponent"] = base_figure_dict.copy()

    if st.session_state.get("parameters_chosen"):
        with st.form("bifurcation diagram"):
            chosen_variable = st.selectbox("Which variable", range(1, 1 + num_equations))
            bif_button = st.form_submit_button("Selected Variable")

        if bif_button:
            if st.session_state["bifurcation_diagrams"].get(chosen_variable) is not None:
                st.pyplot(st.session_state["bifurcation_diagrams"][chosen_variable])
            else:
                chaotic_map = st.session_state["chaotic_map"]
                init_cond = st.session_state["initial_condition"]
                params = st.session_state["parameters"]
                fig = plt.figure()
                d = chaotic_map.bifurcation_dict(init_cond, params, num_points=300)
                for var in [chosen_variable]:
                    st.markdown(f"Bifurcation diagram {var}")
                    fig = plt.figure()
                    x_points = []
                    y_points = []
                    for pam, bif_points in d.items():
                        points = [point[var-1] for point in bif_points]
                        n = len(points)
                        x_points.extend(n * [pam])
                        y_points.extend(points)
                    plt.scatter(x_points, y_points, c='k', s=0.1)
                    plt.ylim(min(y_points) - 0.05 * abs(max(y_points)), max(y_points) + 0.05 * abs(max(y_points)))
                    plt.ylabel(st.session_state.get('variable_names')[var-1], rotation=0)
                    plt.xlabel(st.session_state.get('parameter_names')[which_parameter-1])
                    plt.tight_layout()
                    plt.autoscale(enable=True, axis='x', tight=True)
                    st.session_state["bifurcation_diagrams"][chosen_variable] = fig
                    st.pyplot(fig)

        with st.form("Lyapunov Exponent Diagram"):
            le_chosen_variable = st.selectbox("Which variable", range(1, 1 + num_equations))
            le_button = st.form_submit_button("Selected Variable")

        if le_button:
            if st.session_state["lyapunov_exponent"].get(le_chosen_variable) is not None:
                st.pyplot(st.session_state["lyapunov_exponent"][le_chosen_variable])
            else:
                chaotic_map = st.session_state["chaotic_map"]
                init_cond = st.session_state["initial_condition"]
                params = st.session_state["parameters"]
                fig = plt.figure()
                d = chaotic_map.lyapunov_exponent_dict(init_cond, params)
                for var in [chosen_variable]:
                    st.markdown(f"Lyapunov Exponent Diagram {var}")
                    fig = plt.figure()
                    le_vals = [v[le_chosen_variable - 1] for v in d.values()]
                    plt.plot(d.keys(), le_vals)
                    plt.autoscale(enable=True, axis='x', tight=True)
                    plt.ylabel(st.session_state.get('variable_names')[var-1], rotation=0)
                    plt.ylim(-0.5, np.ceil(max(le_vals) + 0.05 * abs(max(le_vals))))
                    plt.axhline(y=0, c='g', linestyle='--')
                    plt.xlabel(st.session_state.get('parameter_names')[which_parameter-1])
                    plt.tight_layout()

                    st.session_state["lyapunov_exponent"][le_chosen_variable] = fig
                    st.pyplot(fig)


    if st.session_state.params_submitted:
        with st.form("lyapunov exponent"):
            chosen_variable = st.selectbox("Which variable", range(1, 1 + num_equations))
            #fixme This should be a list from multiselection
            le_button = st.form_submit_button("Select Variable")

        if le_button:
            if st.session_state["lyapunov_exponent"].get(chosen_variable) is not None:
                st.pyplot(st.session_state["lyapunov_exponent"][chosen_variable])
            else:
                chaotic_map = st.session_state["chaotic_map"]
                init_cond = st.session_state["initial_condition"]
                params = st.session_state["parameters"]
                fig = plt.figure()
                le_d = chaotic_map.lyapunov_exponent_dict(init_cond, params)
                for var in [chosen_variable]:
                    st.markdown(f"Lyapunov exponent diagram {var}")
                    fig = plt.figure()
                    for pam in le_d:
                        le_d[pam] = le_d[pam][var-1]

                    st.session_state["lyapunov_exponent"][chosen_variable] = fig
                    st.pyplot(fig)


st.header("Select map")

active_tab = st.radio("Options", ["Select map", "Trajectory Analysis", "Dynamical Analysis"])



if active_tab == "Select map":
    select_map()
elif active_tab == "Trajectory Analysis":
    trajectories()
elif active_tab == "Dynamical Analysis":
    dynamical_analysis()




