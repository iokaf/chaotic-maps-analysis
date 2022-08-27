
import streamlit as st
from app_methods import select_map, trajectories, dynamical_analysis, about


with st.sidebar:
    st.header("Select map")

    active_tab = st.radio("Modes", ["Select map", "Trajectory Analysis", "Dynamical Analysis", "About"])

    if st.button("Reset"):
        st.session_state = {}
        active_tab = "Select map"
        st.experimental_rerun()

if active_tab == "Select map":
    select_map()
elif active_tab == "Trajectory Analysis":
    trajectories()
elif active_tab == "Dynamical Analysis":
    dynamical_analysis()
elif active_tab == "About":
    about()
