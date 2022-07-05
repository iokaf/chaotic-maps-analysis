import os
import sys
import numpy as np
from pathlib import Path
import importlib.util
import streamlit as st
import matplotlib.pyplot as plt

from Discrete_Chaos.DiscreteChaos import DiscreteChaosSuite as DCS


def select_map():
    st.session_state["iteration"] = None

    with st.form("Map selection"):
        tab_space = 4 * " "
        d = f"def iteration(x, r): \n {tab_space}x, = x \n {tab_space} r, = r \n {tab_space} x_new = r * x * (1 - x) \n {tab_space}return x_new,"


        st.markdown("# Map Selection")

        iteration_function = st.text_area("Type the function here", height = 200)

        num_equations = st.number_input("System Dimension", min_value = 1, step = 1)
        num_parameters = st.number_input("Number of parameters", min_value = 1, step = 1)

        select_map = st.form_submit_button("Select map")
        
    if select_map:

        base_figure_dict = dict(zip(range(num_equations), num_equations * [None]))
        st.session_state["bifurcation_diagrams"] = base_figure_dict.copy()
        st.session_state["lyapunov_exponent"] = base_figure_dict.copy()

        st.session_state.params_submitted = False
        curr_dir = os.getcwd()
        file_name = curr_dir + "/temp.py"
        k = 0
        while os.path.exists(file_name):
            file_name =  f"{curr_dir}/temp_{k}.py"
            k += 1


        with open(file_name, "w") as file:
            st.write(os.path.exists(file_name))
            st.write(file_name)
            file.write(iteration_function)

        path = Path(file_name)

        spec = importlib.util.spec_from_file_location(path.stem, path)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = foo
        spec.loader.exec_module(foo)

        iteration = foo.iteration

        st.session_state["iteration"] = iteration
        st.write(iteration)
        os.remove(file_name)

        
        iteration = st.session_state["iteration"]
        chaotic_map = DCS(iteration)
        st.session_state["chaotic_map"] = chaotic_map