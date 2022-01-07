import numpy as np 
import yfinance as yf
import streamlit as st 
import matplotlib.pyplot as plt 

from fbprophet import Prophet #pacote machine propphet
from fbprophet.plot import plot_plotly #pacote de machine
from plotly import graph_objs as go #criar graficos apartir do facebookproft
from datetime import date #data
#import wanings 
#warnings.filterwarnings("ignore")


#Definir a data de inicio para coleta de dados

INICIO = "2015-01-01"

#DEfine a data de fim para coleta de dados(data de hoje, execucação do SCript)

#vai pegar a data atual do computador e transforma emdia , mes e ano 
HOJE = date.today().strftime("%Y-%m-%d")



#EXTRAIR OS DADOS

#Definir o tirulo do DASH

st.title('Análise de Ações e Previsões')
st.title('Análise Interativo e em Tempo Real Para Previsões de Ativos Financeiros')


st.subheader('Clauder Noronha')

#Define o código das empresas para coleta dos dados de ativos financeiros

#https://finance.yahoo.com/most-active

empresas = ('PBR','GOOG','UBER','PFE')

#Define de qual empresa usaremos os dados por vez
#caixa de seleção para o usuario
empresa_selecionada = st.selectbox('Selecione a Empresa Para as Previsões de Ativos Financeiros:', empresas)


#Extrair os dados com uma função 

@st.cache# serve para quando extrair os dados para armazenar no cache
def carrega_dados(ticker):
	dados = yf.download(ticker, INICIO, HOJE)#chamar o pacote do yahoo, fazer o download do ticker que é o range dos dados do inicio até hoje 
	dados.reset_index(inplace=True)#zerar e criar um index apartir do zero, 
	return dados



#Mensagem de carga dos dados 

mensagem = st.text('Carregando os dados...')

#carregar os dados
dados = carrega_dados(empresa_selecionada)


#Mensagem de encerramento da carga dos daos
mensagem.text('Carregando os dados...Concluído!')


#Sub-tirulos
st.subheader('Visualização dos Dados Brutos')
st.write(dados.tail())#pegar a parte final do dataset e colocar



#CRIAÇÃO DE GRAFICOS

###Função para o PLOT dos DADOS Brutos

def plot_dados_brutos():
	fig = go.Figure()#Cria uma figura
	fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Open'], name ='stock-open'))
	fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Close'], name = 'stock_close'))
	fig.layout.update(title_text = 'Preço de Abertura e Fechamento das Ações', xaxis_rangeslider_visible = True)#Visiblidade 
	st.plotly_chart(fig)#Chamar o grafico com a função plotly

#EXECUTA A FUNÇÃO 

plot_dados_brutos()



#CRIANDO O MODELODE PREVISÃO DE AÇÕES USANDO O MACHINE


st.subheader('Previsões com Machine Learning')

# Prepara os dados para as previsões com o pacote Prophet
df_treino = dados[['Date','Close']]
df_treino = df_treino.rename(columns = {"Date": "ds", "Close": "y"})

# Cria o modelo
modelo = Prophet()




#MOSTAR AS PREVISOES 


# Treina o modelo
modelo.fit(df_treino)

# Define o horizonte de previsão
#É recomendavel colocar um periodo de no máximo 4 anos, passa disso vira chute 
num_anos = st.slider('Horizonte de Previsão (em anos):', 1, 4)

# Calcula o período em dias
periodo = num_anos * 365

# Prepara as datas futuras para as previsões
futuro = modelo.make_future_dataframe(periods = periodo)

# Faz as previsões
forecast = modelo.predict(futuro)

# Sub-título
st.subheader('Dados Previstos')

# Dados previstos
st.write(forecast.tail())
    
# Título
st.subheader('Previsão de Preço dos Ativos Financeiros Para o Período Selecionado')

# Plot
grafico2 = plot_plotly(modelo, forecast)
st.plotly_chart(grafico2)


#No grafico vai aparecer uns pontos pretos que são a data de hoje . A linha
#azul é a previsão e a linha azul sombreada reprsenta o intervalo de confiança 











st.subheader('Clauder Noronha')
st.subheader('Brasília - 07-01-2022')


