import streamlit as st
import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px
from sklearn import linear_model
import statsmodels.api as sm

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

def p2f(x):
    if x == '':
        return None
    else:
        return float(x.rstrip('%'))/100

df = pd.read_csv('https://raw.githubusercontent.com/zapetras97/HealthEquity/main/Data/Tabular%20data%20table.csv', 
                 header = 0, 
                 names = ["FIPS", "County", "TOT_POP", "DeathsPerCase", "DeathRate", "CaseRate", "UninsuredRate", 
                          "WAC", "BAC", "H", "IAC", "AAC", "NAC"], 
                 converters={'DeathsPerCase':p2f, "UninsuredRate":p2f, "WAC":p2f, "BAC":p2f, "H":p2f, "IAC":p2f, 
                             "AAC":p2f, "NAC":p2f}, 
                 dtype = {"FIPS": str})

fig1 = px.choropleth(df, geojson=counties, locations='FIPS', color='DeathsPerCase',
                           color_continuous_scale="OrRd",
                           range_color=(0.1886, 0.481),
                           scope="usa",
                           labels={'DeathsPerCase':'Deaths per New Case Rate'},
                           hover_name = 'County',
                          )
fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig1)

fig2 = px.choropleth(df, geojson=counties, locations='FIPS', color='CaseRate',
                           color_continuous_scale="OrRd",
                           range_color=(142.2, 1204),
                           scope="usa",
                           labels={'CaseRate':'Age-Adjusted New Case Rate'},
                           hover_name = 'County',
                          )
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig2)

DeathsPerCaseEthnicities = df[["DeathsPerCase", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X = DeathsPerCaseEthnicities[["WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y = DeathsPerCaseEthnicities["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
st.write(est2.summary())