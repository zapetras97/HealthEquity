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
                           hovertemplate = 'Deaths Per Case: %{DeathsPerCase*100}%' + 'Age-Adjusted New Case Rate: %{CaseRate}' + 'Uninsured Rate: %{UninsuredRate}'
                           
                           #hover_data={'FIPS': False, 'DeathsPerCase':'Deaths Per Case', 'CaseRate':'Age-Adjusted New Case Rate', 'Uninsured Rate' : 'UninsuredRate',
                                       #'White Alone or in Combination':'WAC', 'Black or African American Alone or in Combination':'BAC', 
                                       #'American Indian and Alaska Native Alone or in Combination': 'IAC', 'Asian Alone or in Combination': 'AAC', 
                                       #'Native Hawaiian and Other Pacific Islander Alone or in Combination': 'NAC'}
                          )
fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig1)

fig2 = px.choropleth(df, geojson=counties, locations='FIPS', color='DeathRate',
                           color_continuous_scale="OrRd",
                           range_color=(58.4, 387.9),
                           scope="usa",
                           labels={'DeathRate':'Age-Adjusted Death Rate'},
                           hover_name = 'County',
                          )
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig2)

fig3 = px.choropleth(df, geojson=counties, locations='FIPS', color='CaseRate',
                           color_continuous_scale="OrRd",
                           range_color=(142.2, 659.2),
                           scope="usa",
                           labels={'CaseRate':'Age-Adjusted New Case Rate'},
                           hover_name = 'County',
                          )
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig3)

fig4 = px.choropleth(df, geojson=counties, locations='FIPS', color='UninsuredRate',
                           color_continuous_scale="OrRd",
                           range_color=(0.0144, 0.3245),
                           scope="usa",
                           labels={'UninsuredRate':'Rate of Uninsured Population'},
                           hover_name = 'County',
                          )
fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig4)

DeathsPerCaseEthnicities = df[["DeathsPerCase", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X1 = DeathsPerCaseEthnicities[["WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y1 = DeathsPerCaseEthnicities["DeathsPerCase"]

regr1 = linear_model.LinearRegression()
regr1.fit(X1, y1)

X2_1 = sm.add_constant(X1)
est1_1 = sm.OLS(y1, X2_1)
est2_1 = est1_1.fit()
st.write(est2_1.summary())

DPCEthnicitiesUninsured = df[["DeathsPerCase", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X2 = DPCEthnicitiesUninsured[["UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y2 = DPCEthnicitiesUninsured["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X2, y2)

X2_2 = sm.add_constant(X2)
est1_2 = sm.OLS(y2, X2_2)
est2_2 = est1_2.fit()
st.write(est2_2.summary())