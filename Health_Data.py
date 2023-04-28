# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:36:57 2023

@author: zanepetras
"""

import streamlit as st
import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

def p2f(x):
    if x == '':
        return None
    else:
        return float(x.rstrip('%'))/100

df = pd.read_csv('Data\Tabular data table.csv', header = 0, names = ["FIPS", "County", "TOT_POP", "DeathsPerCase", "DeathRate", "CaseRate", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"], converters={'DeathsPerCase':p2f, "UninsuredRate":p2f, "WAC":p2f, "BAC":p2f, "H":p2f, "IAC":p2f, "AAC":p2f, "NAC":p2f}, dtype = {"FIPS": str})
deaths = df[["FIPS", "DeathsPerCase"]]
fig = px.choropleth(deaths, geojson=counties, locations='FIPS', color='DeathsPerCase',
                           color_continuous_scale="Viridis",
                           range_color=(0.1886, 0.481),
                           scope="usa",
                           labels={'DeathsPerCase':'Deaths per New Case Rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig)