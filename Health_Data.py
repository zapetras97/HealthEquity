import streamlit as st
import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px
from sklearn import linear_model
import statsmodels.api as sm

#Load Data

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

#Create Maps

fig1 = px.choropleth(df, geojson=counties, locations='FIPS', color='DeathsPerCase',
                     color_continuous_scale="YlOrRd",
                     range_color=(0.1886, 0.481),
                     scope="usa",
                     labels={'DeathsPerCase':'Deaths per New Case Rate', 'UninsuredRate': 'Uninsured Rate'},
                     hover_name = 'County',
                     hover_data={'FIPS': False, 'TOT_POP':True, 'DeathsPerCase':':.3f', 'CaseRate':':.3f', 'UninsuredRate' : ':.3f', 'WAC':':.3f', 'BAC':':.3f', 
                                 'IAC':':.3f', 'AAC':':.3f', 'NAC':':.3f'},
                     width = 1200,
                     height = 600
                    )
fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#st.write(fig1)


fig2 = px.choropleth(df, geojson=counties, locations='FIPS', color='DeathRate',
                     color_continuous_scale="YlOrRd",
                     range_color=(58.4, 387.9),
                     scope="usa",
                     labels={'DeathRate':'Age-Adjusted Death Rate'},
                     hover_name = 'County',
                     hover_data={'FIPS': False, 'DeathsPerCase':':.3f', 'CaseRate':':.3f', 'UninsuredRate' : ':.3f', 'WAC':':.3f', 'BAC':':.3f', 'IAC':':.3f',
                                 'AAC':':.3f', 'NAC':':.3f'}
                    )
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#st.write(fig2)

fig3 = px.choropleth(df, geojson=counties, locations='FIPS', color='CaseRate',
                     color_continuous_scale="YlOrRd",
                     range_color=(142.2, 659.2),
                     scope="usa",
                     labels={'CaseRate':'Age-Adjusted New Case Rate'},
                     hover_name = 'County',
                     hover_data={'FIPS': False, 'DeathsPerCase':':.3f', 'CaseRate':':.3f', 'UninsuredRate' : ':.3f', 'WAC':':.3f', 'BAC':':.3f', 'IAC':':.3f',
                                 'AAC':':.3f', 'NAC':':.3f'}
                    )
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#st.write(fig3)

fig4 = px.choropleth(df, geojson=counties, locations='FIPS', color='UninsuredRate',
                     color_continuous_scale="YlOrRd",
                     range_color=(0.0144, 0.3245),
                     scope="usa",
                     labels={'UninsuredRate':'Rate of Uninsured Population'},
                     hover_name = 'County',
                     hover_data={'FIPS': False, 'DeathsPerCase':':.3f', 'CaseRate':':.3f', 'UninsuredRate' : ':.3f', 'WAC':':.3f', 'BAC':':.3f', 'IAC':':.3f',
                                'AAC':':.3f', 'NAC':':.3f'}
                          )
fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#st.write(fig4)

#Regression Analysis

# DPC, no extra variables

DPC = df[["DeathsPerCase", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DPC = DPC[["WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DPC = DPC["DeathsPerCase"]

regr_DPC = linear_model.LinearRegression()
regr_DPC.fit(X_DPC, y_DPC)

X2_DPC = sm.add_constant(X_DPC)
est1_DPC = sm.OLS(y_DPC, X2_DPC)
est2_DPC = est1_DPC.fit()

# DPC with Uninsured rate

DPC_U = df[["DeathsPerCase", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DPC_U = DPC_U[["UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DPC_U = DPC_U["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_DPC_U, y_DPC_U)

X2_DPC_U = sm.add_constant(X_DPC_U)
est1_DPC_U = sm.OLS(y_DPC_U, X2_DPC_U)
est2_DPC_U = est1_DPC_U.fit()

# DPC with population

DPC_P = df[["DeathsPerCase", "TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DPC_P = DPC_P[["TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DPC_P = DPC_P["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_DPC_P, y_DPC_P)

X2_DPC_P = sm.add_constant(X_DPC_P)
est1_DPC_P = sm.OLS(y_DPC_P, X2_DPC_P)
est2_DPC_P = est1_DPC_P.fit()

# DPC with uninsured rate and population

DPC_PU = df[["DeathsPerCase", "TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DPC_PU = DPC_PU[["TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DPC_PU = DPC_PU["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_DPC_PU, y_DPC_PU)

X2_DPC_PU = sm.add_constant(X_DPC_PU)
est1_DPC_PU = sm.OLS(y_DPC_PU, X2_DPC_PU)
est2_DPC_PU = est1_DPC_PU.fit()

# DR, no extra variables

DR = df[["DeathRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DR = DR[["WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DR = DR["DeathRate"]

regr_DR = linear_model.LinearRegression()
regr_DR.fit(X_DR, y_DR)

X2_DR = sm.add_constant(X_DR)
est1_DR = sm.OLS(y_DR, X2_DR)
est2_DR = est1_DR.fit()

# DR with Uninsured rate

DR_U = df[["DeathRate", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DR_U = DR_U[["UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DR_U = DR_U["DeathRate"]

regr = linear_model.LinearRegression()
regr.fit(X_DR_U, y_DR_U)

X2_DR_U = sm.add_constant(X_DR_U)
est1_DR_U = sm.OLS(y_DR_U, X2_DR_U)
est2_DR_U = est1_DR_U.fit()

# DR with population

DR_P = df[["DeathRate", "TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DR_P = DR_P[["TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DR_P = DR_P["DeathRate"]

regr = linear_model.LinearRegression()
regr.fit(X_DR_P, y_DR_P)

X2_DR_P = sm.add_constant(X_DR_P)
est1_DR_P = sm.OLS(y_DR_P, X2_DR_P)
est2_DR_P = est1_DR_P.fit()

# DR with uninsured rate and population

DR_PU = df[["DeathRate", "TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_DR_PU = DR_PU[["TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_DR_PU = DR_PU["DeathRate"]

regr = linear_model.LinearRegression()
regr.fit(X_DR_PU, y_DR_PU)

X2_DR_PU = sm.add_constant(X_DR_PU)
est1_DR_PU = sm.OLS(y_DR_PU, X2_DR_PU)
est2_DR_PU = est1_DR_PU.fit()

# NC, no extra variables

NC = df[["DeathsPerCase", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_NC = NC[["WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_NC = NC["DeathsPerCase"]

regr_NC = linear_model.LinearRegression()
regr_NC.fit(X_NC, y_NC)

X2_NC = sm.add_constant(X_NC)
est1_NC = sm.OLS(y_NC, X2_NC)
est2_NC = est1_NC.fit()

# NC with Uninsured rate

NC_U = df[["DeathsPerCase", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_NC_U = NC_U[["UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_NC_U = NC_U["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_NC_U, y_NC_U)

X2_NC_U = sm.add_constant(X_NC_U)
est1_NC_U = sm.OLS(y_NC_U, X2_NC_U)
est2_NC_U = est1_NC_U.fit()

# NC with population

NC_P = df[["DeathsPerCase", "TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_NC_P = NC_P[["TOT_POP", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_NC_P = NC_P["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_NC_P, y_NC_P)

X2_NC_P = sm.add_constant(X_NC_P)
est1_NC_P = sm.OLS(y_NC_P, X2_NC_P)
est2_NC_P = est1_NC_P.fit()

# NC with uninsured rate and population

NC_PU = df[["DeathsPerCase", "TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]].dropna()
X_NC_PU = NC_PU[["TOT_POP", "UninsuredRate", "WAC", "BAC", "H", "IAC", "AAC", "NAC"]]
y_NC_PU = NC_PU["DeathsPerCase"]

regr = linear_model.LinearRegression()
regr.fit(X_NC_PU, y_NC_PU)

X2_NC_PU = sm.add_constant(X_NC_PU)
est1_NC_PU = sm.OLS(y_NC_PU, X2_NC_PU)
est2_NC_PU = est1_NC_PU.fit()



#sStreamlit Setup

st.set_page_config(layout="wide")

with st.sidebar:
    mapToDisplay = st.radio("Select which map to display", ('Deaths Per Case', 'Death Rate', 'New Case Rate', 'Uninsured Rate'))
    regressionVar = st.radio("Select regression analysis Y variable", ('Deaths Per Case', 'Death Rate', 'New Case Rate'))
    st.write("Additional explanatory variables:")
    uninsured = st.checkbox('Uninsured Rate')
    population = st.checkbox('Population')
    

if mapToDisplay == 'Deaths Per Case':
    st.title("Deaths per New Case Rate by County")
    st.write(fig1)
elif mapToDisplay == 'Death Rate':
    st.title("Age-Adjusted Deaths per 100,000 Case Rate by County")
    st.write(fig2)
elif mapToDisplay == 'New Case Rate':
    st.title("Age-Adjusted New Cases per 100,000 Case Rate by County")
    st.write(fig3)
else:
    st.title("Rate of Uninsured Population by County")
    st.write(fig4)
    
if regressionVar == 'Deaths Per Case':
    if uninsured and population:
        st.write(est2_DPC_PU.summary())
    elif uninsured:
        st.write(est2_DPC_U.summary())
    elif population:
        st.write(est2_DPC_P.summary())
    else:
        st.write(est2_DPC.summary())
        
elif regressionVar == 'Death Rate':
    if uninsured and population:
        st.write(est2_DR_PU.summary())
    elif uninsured:
        st.write(est2_DR_U.summary())
    elif population:
        st.write(est2_DR_P.summary())
    else:
        st.write(est2_DR.summary())
else:
    if uninsured and population:
        st.write(est2_NC_PU.summary())
    elif uninsured:
        st.write(est2_NC_U.summary())
    elif population:
        st.write(est2_NC_P.summary())
    else:
        st.write(est2_NC.summary())



