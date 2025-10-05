import streamlit as st

import camelot

from pandas import DataFrame,read_csv,to_datetime,concat
import numpy

from matplotlib import pyplot
from seaborn import countplot,barplot,heatmap

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

import joblib

def transform():
  arquivos = ['matriculas-ativas.pdf','matriculas-canceladas.pdf']

  data=[]
  for arquivo in arquivos:
    path = arquivo
    dfs=[]

    stream = camelot.read_pdf(path,pages='all',flavor='stream')
    stream[0].df = stream[0].df.iloc[2:]
    stream[-1].df = stream[-1].df.iloc[:-1]

    for table in stream:
      df = table.df
      df.columns = df.iloc[0]
      df = df.iloc[1:]
      dfs.append(df)

    df = concat(dfs).reset_index(drop=True)

    data.append(df)

  df = concat(data).reset_index(drop=True)
  df.to_csv('data.csv',sep=';',index=False)

  return


def load_data(transform=True):
    try:
        df = read_csv('data.csv',sep=';',decimal=',')
    except:
        transform()
        df = read_csv('data.csv',sep=';',decimal=',')
  
    if transform:
        ativas = df[df['Situação']=='Ativa']
        canceladas = df[df['Situação']=='Cancelada'].sample(n=340)
        df = concat([ativas,canceladas])
        #df = read_csv('data.csv',sep=';',decimal=',')
        df = df.drop(['Código','Nome Completo','Fim'], axis=1)
  
        values = df['Plano'].value_counts().index
        dicio = dict(zip(values,range(len(values))))
        df['Plano'] = df['Plano'].map(dicio)

        df['Situação'] = df['Situação'].map({'Ativa':0,'Cancelada':1})
        df['Mes'] = to_datetime(df['Início'],dayfirst=True).dt.month
        #df['Ano'] = to_datetime(df['Início'],dayfirst=True).dt.year
        df = df.drop('Início',axis=1)

        df = df.fillna(-1)
        df = df.astype('float64')
  
    return df

def create_model():
  df = load_data()

  xtrain = df.drop('Situação',axis=1)
  ytrain = df['Situação']

  model = DecisionTreeClassifier()
  model.fit(xtrain,ytrain)

  joblib.dump(model,'model.pkl.z')
   
  return

def predict_new_input(plano,valor_mensal,mes):
  model = joblib.load('model.pkl.z')
  input = DataFrame({'Plano':[plano],'Valor Mensal':[valor_mensal],'Mes':[mes]})
  yproba = model.predict_proba(input)

  return yproba


st.title('Dashboard')


tab1, tab2 = st.tabs(['Informações','Modelo Preditivo'])
with tab1:
    
    df = load_data(transform=False)

    container = st.container()
    a,b = container.columns(2)

    a.write('Porcentagem de Matrículas Canceladas')
    data = 100*df['Situação'].value_counts()/len(df)
    container = st.container()
    fig1,ax1 = pyplot.subplots()
    ax1.pie(data.values,labels=data.index,autopct='%1.f%%')
    a.pyplot(fig1)
    
    b.write('Valor Mensal')
    fig1,ax1 = pyplot.subplots()
    names = df['Situação'].value_counts().index
    for name in names:
        ax1.hist(df[df['Situação']==name]['Valor Mensal'],alpha=0.5,label=name)
    ax1.legend()
    b.pyplot(fig1)
 
    container = st.container()
    a,b = container.columns(2)

    a.write('Os cinco planos mais populares')

    filtrado = df[df['Plano'].isin(df['Plano'].value_counts().sort_values(ascending=False).head().index.to_list())]
    fig2,ax2 = pyplot.subplots()
    countplot(data=filtrado,x='Plano',hue='Situação')
    a.pyplot(fig2)

    b.write('Os cinco planos menos populares')

    filtrado = df[df['Plano'].isin(df['Plano'].value_counts().sort_values(ascending=False).tail().index.to_list())]
    fig2,ax2 = pyplot.subplots()
    countplot(data=filtrado,x='Plano',hue='Situação')
    b.pyplot(fig2)

    container = st.container()
    a,b = container.columns(2)
    
    df['Ano'] = to_datetime(df['Início'],dayfirst=True).dt.year
    df['Mes'] = to_datetime(df['Início'],dayfirst=True).dt.month
    df = df.drop('Início',axis=1)

    a.write('Número de Matrículas por Ano')
    fig1,ax1 = pyplot.subplots()
    countplot(data=df,x='Ano',hue='Situação')
    a.pyplot(fig1)

    b.write('Número de Matrículas por Mês (Todos os Anos)')
    fig1,ax1 = pyplot.subplots()
    countplot(data=df,x='Mes',hue='Situação')
    b.pyplot(fig1)
    
    container = st.container()
    a,b = container.columns(2)

    a.write('Número de Matrículas por Mês em 2024')
    fig1,ax1 = pyplot.subplots()
    dff = df[df['Ano']==2024]
    countplot(data=dff,x='Mes',hue='Situação')
    a.pyplot(fig1)

    b.write('Número de Matrículas por Mês em 2025')
    fig1,ax1 = pyplot.subplots()
    dff = df[df['Ano']==2025]
    countplot(data=dff,x='Mes',hue='Situação')
    b.pyplot(fig1)

    

    
with tab2:
    container = st.container()
    a,b = container.columns(2)

    plano = a.text_input('Plano')
    mes = a.text_input('Mes')
    valor = a.text_input('Valor')
    
    if plano and mes and valor:
        proba = predict_new_input(plano,valor,mes)
        #a.write(proba)
        fig1,ax1 = pyplot.subplots()
        ax1.pie(proba[0],labels=['Ativa','Cancelada'],autopct='%1.f%%')
        b.pyplot(fig1)
