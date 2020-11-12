import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
plt.style.use('seaborn')
parametro_plot=0 # (0:desactivados, 1: activados)

### Usado para línea de comentarios
# En principio de linea, para evitar ejecución

### Fuente: INE, Viajes, pernoctaciones, duración media y gasto por comunidad autónoma de residencia de los viajeros
df=pd.read_csv('12448.csv',delimiter=';')
print ('df.columns:\n',df.columns,sep='')
df.rename(columns={df.columns[0]:'CCAA',\
                   df.columns[1]:'Concepto',\
                   df.columns[2]:'Magnitud',\
                   df.columns[3]:'Periodo',\
                   df.columns[4]:'Valor'},\
                   inplace=True)
for col in df.columns: print (f'df[\'{col}\'].unique():\n',\
                              df[col].unique(),sep='')
### Elimina valores de posición en denominación de CCAA:
df['CCAA'][df['CCAA']!='Total']=df['CCAA'].str[3:]
### Simplifica denominación de algunas CCAA:
df['CCAA'][df['CCAA']==df['CCAA'].unique()[3]]='Asturias'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[4]]='Baleares'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[8]]='Castilla La Mancha'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[13]]='Madrid'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[14]]='Murcia'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[15]]='Navarra'
df['CCAA'][df['CCAA']==df['CCAA'].unique()[17]]='La Rioja'
print ('df[\'CCAA\'].unique():\n',df['CCAA'].unique(),sep='')
### Series indep. de año y periodo para poder modificar después el orden de agrupación:
df['Año']=df['Periodo'].str[:4]
df['Trimestre']=df['Periodo'].str[-2:]
# print ('df[\'Año\'].unique():\n',df['Año'].unique(),sep='')
# print ('df[\'Trimestre\'].unique():\n',df['Trimestre'].unique(),sep='')
def arreglar_num(x):
    """ arreglar_num reconoce celdas vacías o con dos puntos (..) como np.nan,
        elimina puntos de posición de miles y sustituye coma por punto decimal
    """
    if type(x) is not str: y=x
    elif (len(x)==0) | (x=='..'): y=np.nan
    else:
        y=''
        for i in range(len(x.split('.'))): y=y+x.split('.')[i]
        try: y=int(y)
        except: y=float(y.split(',')[0]+'.'+y.split(',')[1])
    return y
help(arreglar_num)
df['Valor']=df['Valor'].apply(arreglar_num)
### Pivot Table de viajes (magnitud absoluta) por CCAA y por periodo:
PT_T_Viajes=df[(df['Magnitud']=='Valor absoluto')&(df['Concepto']=='Viajes')]\
            .pivot_table(index=['Año','Trimestre'],\
                         columns='CCAA',\
                         values='Valor')
print ('PT_T_Viajes:\n',PT_T_Viajes,sep='')
#PT_T_Viajes[PT_T_Viajes.columns[:-1]].plot() # plot sin linea de Total (distorsiona)
print ('Diagramas de cajas y bigotes _________________________________________')
if parametro_plot==1: sns.set_theme(style='whitegrid')
df_sns=PT_T_Viajes.reset_index() # para representar cajas y bigotes, las series relevantes no pueden ser índices
if parametro_plot==1: sns.boxplot(x=df_sns['Trimestre'],y=df_sns['Asturias'])
if parametro_plot==1: plt.show()
df_sns_T1T2=df_sns[(df_sns['Trimestre']=='T1')|(df_sns['Trimestre']=='T2')]
print ('df_sns_T1T2:\n',df_sns_T1T2,sep='')
if parametro_plot==1: sns.boxplot(x=df_sns_T1T2['Año'],y=df_sns_T1T2['Asturias'],palette='Blues')
if parametro_plot==1: plt.show()
print ('______________________________________________________________________')
### Pivot table de proporción de cada CCAA sobre el gasto total en cada periodo:
PT_Distribucion_Viajes=df[(df['Magnitud']=='Distribución porcentual')\
            &(df['Concepto']=='Viajes')]\
            .pivot_table(index=['Año','Trimestre'],\
                         columns='CCAA',\
                         values='Valor')
#print ('PT_Distribucion_Viajes:\n',PT_Distribucion_Viajes,sep='')
### Pivot table de tasas interanuales de variación de viajes por CCAA
PT_VariacionAnual_Viajes=df[(df['Magnitud']=='Variación anual')\
            &(df['Concepto']=='Viajes')]\
            .pivot_table(index=['Año','Trimestre'],\
                         columns='CCAA',\
                         values='Valor')
print ('PT_VariacionAnual_Viajes:\n',PT_VariacionAnual_Viajes,sep='')
print ('Producto cartesiano de valores únicos de dos series __________________')
### df auxiliar para construir índices de otro df
df_Año=pd.DataFrame([df['Año'].unique()],index=['Año']).T\
                    .sort_values(by='Año',ascending=True)
df_Año['Indice']=0 # índices todos nulos para outer merge posterior
df_Año.set_index(['Indice'],inplace=True)
df_Trimestre=pd.DataFrame([df['Trimestre'].unique()],index=['Trimestre']).T\
                          .sort_values(by='Trimestre')
df_Trimestre['Indice']=0 # índices todos nulos para outer merge posterior
df_Trimestre.set_index(['Indice'],inplace=True)
df_Aux=df_Año.merge(df_Trimestre,how='outer',left_index=True,right_index=True)
df_Aux=df_Aux.iloc[4:-2].reset_index()
del df_Aux['Indice'] # prescinde de índices nulos
print ('df_Aux:\n',df_Aux,sep='')
print ('Construcción de df de tasas de variación interanual __________________')
### Construcción de df de tasas interanuales como ejercicio:
L_L_TasasInteranuales_Viajes=list() # lista de listas de tasas para cada col.(CCAA)
for col in PT_T_Viajes:
    L_TasaInteranual_Viajes=[(PT_T_Viajes[col].iloc[i]\
                    -PT_T_Viajes[col].iloc[i-4])\
                    /PT_T_Viajes[col].iloc[i-4]\
                    for i in range(4,PT_T_Viajes.shape[0])]
    L_L_TasasInteranuales_Viajes.append(L_TasaInteranual_Viajes)
df_TasasInteranuales_Viajes=pd.DataFrame(L_L_TasasInteranuales_Viajes,\
                                         index=PT_T_Viajes.columns).T
df_TasasInteranuales_Viajes=df_TasasInteranuales_Viajes*100
df_TasasInteranuales_Viajes=pd.concat([df_Aux,df_TasasInteranuales_Viajes],\
                                      axis=1)
df_TasasInteranuales_Viajes.set_index(['Año','Trimestre'],inplace=True)
print ('df_TasasInteranuales_Viajes:\n',df_TasasInteranuales_Viajes,sep='')
### Comprobación:
df_Check=PT_VariacionAnual_Viajes-df_TasasInteranuales_Viajes
print ('df_Check:\n',df_Check,sep='')
print ('df_Check[df_Check>.005].notna().any():\n',\
        df_Check[df_Check>.005].notna().any(),sep='')
### Cálulo error medio (por redondeo en datos importados):
df_Check.loc['Error medio CCAA']=df_Check[df_Check.columns[:-1]].mean() # excluye col. Total
df_Check.loc['Tamaño muestra CCAA']=df_Check[df_Check.columns[:-1]]\
                    .iloc[:df_TasasInteranuales_Viajes.shape[0]].count() # excluye tb. fila de error medio
df_Check.loc['Error x tamaño CCAA']=df_Check.loc['Error medio CCAA']\
                                    *df_Check.loc['Tamaño muestra CCAA']
df_Check['Sumas (exc.T)']=df_Check[df_Check.columns[:-1]].sum(axis=1) # Suma por filas excluyendo col. Total
df_Check.loc['Error medio total','Sumas (exc.T)']\
        =df_Check.loc['Error x tamaño CCAA','Sumas (exc.T)']\
        /df_Check.loc['Tamaño muestra CCAA','Sumas (exc.T)']
print ('df_Check:\n',df_Check,sep='')
### Para excluir alguna columna intermedia (Aragón):
CCAA_sin_Aragon=[i for i in df_Check.columns[:1]]\
                +[i for i in df_Check.columns[2:19]]
print ('CCAA_sin_Aragon',CCAA_sin_Aragon,sep='')
print ('df_Check[CCAA_sin_Aragon].head():\n',\
        df_Check[CCAA_sin_Aragon].head(),sep='')
print ('Análisis de estacionalidad (H. multiplicativa) _______________________')
df_Viajes=df[(df['CCAA']=='Total') & (df['Concepto']=='Viajes')\
        & (df['Magnitud']=='Valor absoluto')][['Año','Trimestre','Valor']]\
        .sort_values(by=['Año','Trimestre'])
PT_Viajes=df_Viajes.iloc[:-2].pivot_table(index='Año',\
                                            columns='Trimestre',values='Valor') # excluye datos 2020
print ('df_Viajes:\n',df_Viajes,sep='')
print ('PT_Viajes:\n',PT_Viajes,sep='')
df_Viajes['Tendencia5']=df_Viajes['Valor'].iloc[:-2]\
            .rolling(window=[1,2,2,2,1],win_type='boxcar',center=True).mean() # excluye datos 2020
df_Viajes['S_desestacionalizada']=df_Viajes['Valor'].iloc[:-2]\
                                   /df_Viajes['Tendencia5'].iloc[:-2] # excluye datos 2020
print ('df_Viajes:\n',df_Viajes,sep='')
PT_S_desestacionalizada=df_Viajes.pivot_table(index='Año',\
                            columns='Trimestre',values='S_desestacionalizada')
PT_S_desestacionalizada.loc['I_provisionales']=PT_S_desestacionalizada.mean()
media_I_provisionales=PT_S_desestacionalizada.loc['I_provisionales'].mean()
PT_S_desestacionalizada.loc['I_estacionalidad']\
        =PT_S_desestacionalizada.loc['I_provisionales']/media_I_provisionales
print ('PT_S_desestacionalizada:\n',PT_S_desestacionalizada,sep='')
media_I_estacionalidad=PT_S_desestacionalizada.loc['I_estacionalidad'].mean()
print ('media_I_estacionalidad:',media_I_estacionalidad)
### Para representar, equilibramos la escala:
df_Viajes['S_desestacionalizada_print']=df_Viajes['S_desestacionalizada']\
                                        *df_Viajes['Tendencia5'].mean()
print ('df_Viajes:\n',df_Viajes,sep='')
if parametro_plot==1: df_Viajes.set_index(['Año','Trimestre'])\
        [['Valor','Tendencia5','S_desestacionalizada_print']].plot()
if parametro_plot==1: plt.show()
print ('Pruebas ______________________________________________________________')
print (df_Viajes['Trimestre'].nunique())
Trimestres=df_Viajes['Trimestre'].unique()
print (len(Trimestres))
L_Trimestres=['T1','T2']
print (df_Viajes[df_Viajes['Trimestre'].isin(L_Trimestres)])
