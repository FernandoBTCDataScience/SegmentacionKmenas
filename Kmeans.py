# Lectura del data set
import pandas as pd
import numpy as np

bbdd=pd.read_csv('consulta.csv',sep=';',header=0,names=['SECTOR','CANASTA','CATEGORIA','SUBCATEGORIA','FABRICANTE','AÑO','MES','IMPORTE','UNIDADES','TICKET'])
#print(bbdd[['SECTOR','AÑO','MES']])
#print(bbdd.iloc[0:5,0:2])

#bbdd.fillna(0,inplace=True)
#print(bbdd.loc[0:10,['TICKET']])

#CREACION DEL DATA SET

MACT2=bbdd[bbdd['MES']=='2024-09-01'][['SUBCATEGORIA','IMPORTE','UNIDADES','TICKET']]
MACT2['IMPORTE'] = pd.to_numeric(MACT2['IMPORTE'], errors='coerce')
MACT2['UNIDADES'] = pd.to_numeric(MACT2['UNIDADES'], errors='coerce')
#print(MACT2.dtypes)
MACT=MACT2.groupby('SUBCATEGORIA').agg({'TICKET':'sum','IMPORTE':'sum','UNIDADES':'sum'}).reset_index()
#print(MACT)

MMAA2=bbdd[bbdd['MES']=='2023-09-01'][['SUBCATEGORIA','IMPORTE','UNIDADES','TICKET']]
MMAA2['IMPORTE'] = pd.to_numeric(MMAA2['IMPORTE'], errors='coerce')
MMAA2['UNIDADES'] = pd.to_numeric(MMAA2['UNIDADES'], errors='coerce')
MMAA=MMAA2.groupby('SUBCATEGORIA').agg({'TICKET':'sum','IMPORTE':'sum','UNIDADES':'sum'}).reset_index()

df=MACT.merge(MMAA,on='SUBCATEGORIA')
df['VAR TICKET MMAA']=((df['TICKET_x']/df['TICKET_y'])-1)*100
df['VAR UNIDADES MMAA']=((df['UNIDADES_x']/df['UNIDADES_y'])-1)*100

df['VAR TICKET MMAA'] = pd.to_numeric(df['VAR TICKET MMAA'], errors='coerce')
df['VAR UNIDADES MMAA'] = pd.to_numeric(df['VAR UNIDADES MMAA'], errors='coerce')
df[['VAR TICKET MMAA', 'VAR UNIDADES MMAA']] = df[['VAR TICKET MMAA', 'VAR UNIDADES MMAA']].replace([np.inf, -np.inf], np.nan)
df[['VAR TICKET MMAA', 'VAR UNIDADES MMAA']] = df[['VAR TICKET MMAA', 'VAR UNIDADES MMAA']].fillna(0)
df = df.dropna(subset=['VAR TICKET MMAA', 'VAR UNIDADES MMAA'])

#print(df[['SUBCATEGORIA','IMPORTE_x','IMPORTE_y']])
#print(df)

#Limpieza de datos

from decimal import Decimal, getcontext

Tick_prom=df['VAR TICKET MMAA'].mean()
Unid_prom=df['VAR UNIDADES MMAA'].mean()

limite_superior_Tick = Tick_prom * 2.0
limite_inferior_Tick = Tick_prom * -2.0

limite_superior_unidades = Unid_prom * 2.0
limite_inferior_unidades = Unid_prom * -2.0

df = df[
    (df['VAR TICKET MMAA'] <= limite_superior_Tick) & 
    (df['VAR TICKET MMAA'] >= limite_inferior_Tick) & 
    (df['VAR UNIDADES MMAA'] <= limite_superior_unidades) & 
    (df['VAR UNIDADES MMAA'] >= limite_inferior_unidades)
]

#KMEANS

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

cluster=KMeans(n_clusters=4).fit(df[['VAR TICKET MMAA','VAR UNIDADES MMAA']])
df['Cluster']=cluster.labels_
print(df)
df.to_csv('ncluster.csv', index=False)

#Graficar (revicion)

plt.scatter(df['VAR TICKET MMAA'].astype(float), df['VAR UNIDADES MMAA'].astype(float))
plt.xlabel('VAR TICKET')
plt.ylabel('VAR UNIDADES')
plt.title('Segmentación de Productos')
plt.colorbar(label='Cluster')
plt.show()