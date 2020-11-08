import numpy as np
import pandas as pd

print ('_____pd.DataFrame: ___________________________________________________')
s1=pd.Series([1,5,np.nan,4,7,4,np.nan,4,2,1,3,5,7,4,np.nan,2],\
            index=['v'+str(i) for i in range(1,17)])
s2=pd.Series('ana lola juan lucas lola inés elsa bea ana juan lola'.split(),\
            index=['v'+str(i) for i in range(1,12)])
s3=pd.Series([15,12,15,np.nan,15,12,11,16,15,16,17,11,16,12,np.nan,16],\
            index=['v'+str(i) for i in range(1,17)])
s4=pd.Series([117,np.nan,np.nan,118,115,112,111,np.nan,\
            115,116,117,111,116,112,np.nan,np.nan],\
            index=['v'+str(i) for i in range(1,17)])
df=pd.DataFrame([s1,s2,s3,s4],index=['serie'+str(i) for i in range(1,5)]).T
print ('df:\n',df,sep='') # DataFrame a partir de pd.series. Transpuesta (.T) para que las series sean columnas
print ('df.head():\n',df.head(),sep='') # 5 primeras filas
print ('df.head(3):\n',df.head(3),sep='') # n primeras filas
print ('df.tail():\n',df.tail(),sep='') # 5 últimas filas
print ('df.tail(3):\n',df.tail(3),sep='') # n últimas filas
print ('df.sample():\n',df.sample(),sep='') # una fila aleatoria
print ('df.sample(3):\n',df.sample(3),sep='') # n filas aleatorias
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: índices, cols, orden, valores y valores únicos ____')
print ('df.index:',df.index) # Índices de filas. Es una lista especializada
print ('df.columns:',df.columns) # Nombres de columnas. Es una lista especializada
print ('df.shape:',df.shape) # Orden de la matriz
print ('df.size:',df.size) # no.filas x no.columnas
print ('df.count():\n',df.count(),sep='') # no. de valores informados en cada columna (no incluye Nan)
print ('df.nunique():\n',df.nunique(),sep='') # no. de valores distintos de cada columna (no incluye Nan)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: recuperar columnas, filas y celdas ________________')
print ('df[\'serie3\']:\n',df['serie3'].head(),sep='') # Recuperar una columna
print ('df.loc[\'v5\']:\n',df.loc['v5'],sep='') # Recuperar una fila por índice (.loc[])
print ('df.iloc[4]:\n',df.iloc[4],sep='') # Recuperar una fila por posición (.iloc[])
print ('df.loc[\'v5\',\'serie3\']:',df.loc['v5','serie3']) # Recuperar una celda por fila y columna (.loc[])
print ('df[\'serie3\'].loc[\'v5\']:',df['serie3'].loc['v5']) # Recuperar una celda por índice sobre serie (.loc[])
print ('df[\'serie3\'].iloc[4]:',df['serie3'].iloc[4]) # Recuperar una celda por posición sobre serie (.iloc[])
print ('df.loc[\'v5\'][\'serie3\'] (ineficiente):',df.loc['v5']['serie3']) # Recuperación encadenada: ineficiente
print ('df[\'serie3\'][\'v5\']:',df['serie3']['v5']) # Recuperación encadenada: desaconsejado
print('df[\'serie3\'].iloc[2:5]:\n',df['serie3'].iloc[2:5],sep='') # Recuperar rangos de filas por posición
print('df[\'serie3\'].loc[\'v3\':\'v5\']:\n',df['serie3'].loc['v3':'v5'],sep='') # Recuperar rangos de filas por índice
print ('df[[\'serie1\',\'serie2\']]:\n',df[['serie1','serie2']].head(),sep='') # Recuperar varias columnas con doble corchete [[]]
print ('df.loc[[\'v2\',\'v5\'],[\'serie3\',\'serie4\']]:\n',\
        df.loc[['v2','v5'],['serie3','serie4']],sep='') # Recuperar submatriz por enumeracion de filas y columnas
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: añadir y sobreescribir ____________________________')
print ('df.loc[\'v1\',\'serie2\']:',df.loc['v1','serie2'])
df.loc['v1','serie2']='yoko' # sobreescribe celdas. Permanente
print ('df.loc[\'v1\',\'serie2\']=\'yoko\'; df.loc[\'v1\',\'serie2\']:',\
        df.loc['v1','serie2'])
df.loc['v17']=[5,'esther',19,114] # crea una fila nueva. Permanente
print ('df.loc[\'v17\']=pd.Series([5,\'esther\',19,114]):\n',df.tail(),sep='')
df.loc['v18','serie2']='alba' # crea fila nueva con Nan donde no hay dato
print ('df.loc[\'v18\',\'serie2\']=\'alba\':\n',df.tail(),sep='')
aux='ba bb bc bd be bf bg bh bi bj bk bl bm bn bñ bo bp bq'.split()
df['serie5']=pd.Series(aux,index=['v'+str(i) for i in range(1,len(aux)+1)]) # añade columna nueva hasta longitud del df: si menor, rellena con Nan; si superior, trunca
print ('df[\'serie5\']=col.nueva:\n',df.head(),sep='')
df['serie6']=1001 # Nueva columna con un valor repetido: rellena la serie
print ('df[\'serie6\']=1001:\n',df.head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: añadir con .assing() ______________________________')
print ('df.assign(serie1plus10=df[\'serie1\']+10):\n',\
        df.assign(serie1plus10=df['serie1']+10).head(),sep='') # añade serie. No permanente
print ('df.assign(serie1plus100=lambda s:s[\'serie1\']+100):\n',\
        df.assign(serie1plus100=lambda s:s['serie1']+100).head(),sep='') # expresión alternativa. Preferible para asignaciones múltiples llamando a serie definida en el mismo comando
print ('df.assign(serie6=lambda s:s[\'serie6\']/11):\n',\
        df.assign(serie6=lambda s:s['serie6']/11).head(),sep='') # sobreescribe la columna rebautizada. No permanente
print ('df.assign(serie7=lambda s:s[\'serie2\'][:3],',
        'serie8=lambda s:s[\'serie7\'][1:]):\n',\
        df.assign(serie7=lambda s:s['serie2'][:3],\
        serie8=lambda s:s['serie7'][1:]).head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: eliminar (.drop() y del) __________________________')
print ('df.drop(\'v1\'):\n',df.drop('v1').head(),sep='') # elimina fila (no permantente, admite inplace)
print ('df.drop([\'v5\',\'v6\']):\n',df.drop(['v5','v6']).head(7),sep='') # elimina varias filas (no permanente, admite inplace)
print ('df.drop(\'serie6\',axis=1):\n',df.drop('serie6',axis=1).head(),sep='') # elmina columna (no permanente, admite inplace)
del df['serie5'], df['serie6'] # elimina columnas (permanente)
print ('del df[\'serie5\'], df[\'serie6\']:\n',df.head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: .append() y pd.concat([]) _________________________')
v18=pd.Series([1,'alba',np.nan,130,240],\
              index=[col for col in df.columns]+['serie7'])
v19=pd.Series([2,'pepe',17,117,279],\
              index=[col for col in df.columns]+['serie7'])
df1=pd.DataFrame([v18,v19],index=['v18','v19']) # ATENCIÓN A FILA v18 EN RESULTADOS DE ESTA SECCIÓN
print ('df.append(df1):\n',df.append(df1).tail(),sep='') # Concatena dfs. No permanente (requiere renombrar)
print ('pd.concat([df,df1]):\n',pd.concat([df,df1]).tail(),sep='') # concatena verticalmente (con índices repetidos, repite fila)
print ('pd.concat([df,df1],axis=1):\n',pd.concat([df,df1],axis=1).tail(),sep='') # concatena horizontalmente (con índices repetidos, completa columnas)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: otras transformaciones y operaciones ______________')
print ('df.T:\n',df.T,sep='') # Transposición (no recomendable para cambiar métodos de recuperación de fila y col (hace copia. puede ser impredecible))
df_copia=df.copy()
print ('df_copia=df.copy():\n',df_copia.head(),sep='')
df_reduced=df[['serie1','serie3']]
print ('df_reduced=df[[\'serie1\',\'serie3\']]:\n',df_reduced.head(),sep='') # para seleccionar un subconjunto de columnas
print ('np.array(df.loc[\'v1\']):',np.array(df.loc['v1'])) # convierte fila en vector
print ('np.array(df[\'serie1\']):',np.array(df['serie1'])) # convierte columna en vector
df['serie1']+=20 # Suma un valor a toda la columna. Permanente. Equivalente a df['col']=df['col']+n
print ('df[\'serie1\']+=20:\n',df.head(),sep='')
df['serie1']*=1/2 # Multiplica toda la columna por un valor. Permanente. Equivalente a df['col']=df['col']*n
print ('df[\'serie1\']*=1/2:\n',df.head(),sep='')
print('np.max(df[\'serie1\']):',np.max(df['serie1'])) # valor máximo de una columna
print (np.argmax(df['serie1'])) # (cuidado con óptimos múltiples) posicion del elemento que maximiza una columna. Si hay varios, el primero
print (df['serie1'][df['serie1']==np.max(df['serie1'])]) # conjunto de argumentos que maximizan una columna (usando máscara booleana)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: máscaras booleanas ________________________________')
print ('df:\n',df,sep='')
print ('df[\'serie1\']>=12:\n',(df['serie1']>=12),sep='') # Serie T/F de acuerdo con la condición. Funciona como filtros
print ('df[df[\'serie1\']>=12]:\n',df[df['serie1']>=12],sep='') # Filtro mediante máscara
print ('(df[\'serie1\']>=12).mean():\n',(df['serie1']>=12).mean(),sep='') # la media de una máscara es la proporciòn de trues
print ('df[df[\'serie1\']>=12].mean():\n',df[df['serie1']>=12].mean(),sep='') # media de la serie filtrada
print ('df[(df[\'serie1\']>=12) | (df[\'serie2\']==\'lola\')]:\n',\
        df[(df['serie1']>=12) | (df['serie2']=='lola')].head(),sep='') # df filtrado por máscara booleana
print ('df[(df[\'serie1\']>=12) | (df[\'serie2\']==\'lola\')][\'serie2\']:\n',\
        df[(df['serie1']>=12) | (df['serie2']=='lola')]['serie2'].head(),sep='') # serie filtrada por máscara definida sobre el df
lola= df['serie2']=='lola' # asigna nombre a la máscara
print ('df[\'serie2\']==\'lola\':\n',lola.head(),sep='')
print ('df[~\'lola\']:\n',df[~lola].head(),sep='') # df filtrado por la condición complementaria de otra
print ('df.drop(\'serie2\',axis=1)>=12:\n',\
        (df.drop('serie2',axis=1)>=12).head(),sep='') # máscara sobre un df (excluida columna de strings)
print ('df[df.drop(\'serie2\',axis=1)>=12]:\n',\
        df[df.drop('serie2',axis=1)>=12].head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: modificar índices _________________________________')
df['old_index']=df.index # nueva columma con índices de fila
print ('df[\'old_index\']=df.index:\n',df.head(),sep='')
print('df.reset_index():\n',df.reset_index().head(),sep='') # establece índices númericos. No permanente (admite inplace)
print ('df.rename(index={\'v1\':\'VV1\'}):\n',\
        df.rename(index={'v1':'VV1'}).head(),sep='') # renombra índices de fila individuales. No permanente (admite inplace)
print ('df.rename(columns={\'serie1\':\'SERIE1\'}):\n',\
        df.rename(columns={'serie1':'SERIE1'}).head(),sep='') # renombra columnas individuales. No permanente (admite inplace)
serie_indices=pd.Series(['fila'+str(i) for i in range(1,df.shape[0]+1)]) # serie de índices nuevos
print ('df.set_index(s_index):\n',df.set_index(serie_indices).head(),sep='') # nuevos índices de fila. No permantente (admite inplace)
print ('df.set_index([\'serie2\']):\n',df.set_index(['serie2']).head(),sep='') # Fija la columna indicadas como índice de fila. No permanente  (admite inplace)
dfm=df.set_index(['serie2','serie1']) # Fija las columnas indicadas como índice de fila multinivel
print ('dfm.set_index([\'serie2\',\'serie1\']):\n',dfm.head(),sep='')
print ('dfm.loc[\'lola\',13.5]:\n',dfm.loc['lola',13.5],sep='') # recupera filas con índice multinivel
print ('dfm.loc[[(\'lola\',12.5),(\'lola\',13.5)]]:\n',\
        dfm.loc[[('lola',12.5),('lola',13.5)]],sep='') # recupera varias filas con índice multinivel
dfm.index.names=['nombre','número'] #Cambia el nombre de las columnas que actuan como índices de filas. Permanente
print ('dfm.index.names=[\'nombre\',\'número\']:\n',dfm.head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: importar y exportar csv ___________________________')
"""
df_notas=pd.read_csv('notas.csv',delimiter=';') # to read data from a csv file usando ; como delimitador. Si el archivo está en otra carpeta, ruta
print ('pd.read_csv(\'notas.csv\',delimiter=\';\');\n',df_notas)
df_notas=pd.read_csv('notas.csv',delimiter=';', index_col=0) # fija la primera columna como índices de fila
print ('df_notas=pd.read_csv(\'notas.csv\',delimiter=\';\', index_col=0):\n',\
        df_notas)
df_notas=pd.read_csv('notas.csv',delimiter=';', skiprows=1) # salta n filas y usa la siguiente como nombres de columna
print ('pd.read_csv(\'notas.csv\',delimiter=\';\', skiprows=1):\n',df_notas)
df_notas.to_csv('nuevo_csv')
print ('df_notas.to_csv(\'nuevo.csv\') para exportar a csv')
"""
print ('_____pd.DataFrame: importar y exportar Excel _________________________')
#pd.read_excel('nombre_archivo.xlsx','nombre_hoja').set_index('columna_de_índices') # importa desde read_excel
#with pd.ExcelWriter ('nombre_archivo.xlsx') as writer: # para especificar hoja al exportar
#    df.to_excel(writer, sheet_name='nombre_hoja')
print ('_____pd.DataFrame: .sort_values(by=[\'\']) y .rank() ___________________')
dfs=df.set_index(['serie2'])
del dfs['old_index'] # df simplificado
print ('dfs:\n',dfs.head(),sep='')
print ('dfs.sort_values(by=[\'serie1\',\'serie3\']):\n',\
        dfs.sort_values(by=['serie1','serie3']),sep='') # imprescindible algún criterio de ordenación
print ('dfs.sort_index():\n',dfs.sort_index(),sep='')
print ('dfs[\'serie1\'].rank():\n',dfs['serie1'].rank(),sep='') # resuelve empates con promedio
print ('dfs[\'serie1\'].rank().sort_values():\n',\
        dfs['serie1'].rank().sort_values(),sep='')
print ('dfs[\'serie1\'].rank(method=\'dense\'):\n',\
        dfs['serie1'].rank(method='dense'),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: apply y applymap __________________________________')
# apply sobre pd.Series o sobre pd.DataFrame. Puede ser element-wise o un agregado dependiendo del contexto
# applymap sobre cada elemento (celda) de un pd.DataFrame
# map sobre cada elemento de una pd.Series
import math
dfx=df[['serie1','serie3','serie4']].iloc[0:10] # df solo de números
print ('dfx.head():\n',dfx.head(),sep='')
print ('dfx[\'serie1\'].apply(math.sqrt):\n',\
        dfx['serie1'].apply(math.sqrt).head(),sep='')
print ('dfx.applymap(math.sqrt):\n',dfx.applymap(math.sqrt).head(),sep='')
print ('dfx.head().apply(print):') # EL dataframe serie a serie
dfx.head().apply(print)
print ('dfx.apply(type):\n',dfx.apply(type),sep='')
print ('dfx.applymap(type):\n',dfx.applymap(type).tail(7),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: estadística descriptiva I _________________________')
print ('df:\n',df.head(),sep='')
print ('df.describe():\n',df.describe(),sep='') # en este caso entiende que alguna col. no es numérica: describe() produce count, unique, top y freq
print ('df.nunique():\n',df.nunique(),sep='') # cuenta valores únicos en cada columna (fila unique en .describe())
for col in df.columns:
    print (f'df[\'{col}\'].value_counts().max():',\
           df[col].value_counts().max(),sep='') # la freq del método .describe() (series no numéricas) es la máx de cada columna
for col in df.columns:
    try: df[col]=df[col].astype(float) # asegura tipo float donde es posible
    except: pass
print ('df.describe():\n',df.describe(),sep='') # el resumen .describe() para df numéricos
print ('df.T.describe():\n',df.T.describe(),sep='') # .describe() con series de strings
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: estadística descriptiva II ________________________')
import matplotlib.pyplot as plt
print ('np.sum(df):\n',np.sum(df))
print ('np.sum(df,axis=1):\n',np.sum(df,axis=1).head())
#df.describe().plot()
#plt.title('df.describe().plot()')
df_reduced=df.set_index(['serie2']) # solo series numéricas
del df_reduced['old_index']
print ('df_reduced:\n',df_reduced.tail(7),sep='')
print ('df_reduced.T.describe():\n',df_reduced.T.describe(),sep='') # .describe() solo series numéricas
df_reduced_sinNan=df_reduced[~df_reduced.index.isna()] # omite filas con índice Nan para aplicar .describe().plot()
print ('df_reduced_sinNan:\n',df_reduced_sinNan.tail(),sep='')
print ('df_reduced_sinNan.T.describe():\n',\
        df_reduced_sinNan.T.describe(),sep='')
#df_reduced_sinNan.T.describe().plot() # sobre df sin Nan en índice
#plt.title('df_reduced_sinNan.T.describe().plot()')
#plt.show()
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('____pd.DataFrame: estadística descriptiva III (métodos vectorizados) _')
# los métodos siguientes, sobre df original (con columnas de strings)
print ('df.count():\n',df.count(),sep='')
print ('df.count(axis=1):\n',df.count(axis=1).head(),sep='')
print ('df.nunique():\n',df.nunique(),sep='')
print ('df.nunique(axis)\n:',df.nunique(axis=1).head(),sep='')
print ('df.max():\n',df.max(),sep='') # solo actua sobre col. numéricas
print ('df.max(axis=1):\n',df.max(axis=1).head(),sep='') # solo actua sobre col. numéricas
print ('df.sum():\n',df.sum().head(),sep='') # sobre col. numéricas y strings
print ('df.T:\n',df.T,sep='')
print ('df.sum(axis=1):\n',df.sum(axis=1).head(),sep='') # en filas con num. y strings, omite strings
print ('df.mean():\n',df.mean(),sep='') # solo actua sobre col. numéricas
print ('df.mean(axis=1):\n',df.mean(axis=1).head(),sep='') # no tiene en cuenta valores no numéricos
print ('df.std():\n',df.std().head(),sep='')
print ('df.std(axis=1):\n',df.std(axis=1).head(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: filtrar Nan con .isna(), .isnull() y .notna() _____')
df['serie6']=np.nan
df.loc['v19','serie2']=np.nan
print ('df:\n',df)
print('df[df[\'serie2\'].isna()]:\n',df[df['serie2'].isna()],sep='')
print('df[df[\'serie2\'].isnull()]:\n',df[df['serie2'].isnull()],sep='')
print('df[df[\'serie2\'].notna()]:\n',df[df['serie2'].notna()],sep='')
print ('df.isna():\n',df.isna()) # máscara sobre el df completo: no sirve como filtro
for col in df.columns: print (f'df[\'{col}\'].isna().value_counts():\n',\
                              df[col].isna().value_counts(),sep='')
print ('df.isna().any():\n',df.isna().any(),sep='') # máscara sobre columnas del df: True para columnas con algún Nan
print ('df.isna().all():\n',df.isna().all(),sep='') # máscara sobre columnas del df: True para columnas solo con Nan
print ('df[df.columns[df.isna().all()]]:\n',\
        df[df.columns[df.isna().all()]],sep='') # filtra df.columns con la máscara y muestra las seleccionadas
print ('df[df.columns[df.isna().any()]]:\n',\
        df[df.columns[df.isna().any()]],sep='')
print ('df.T[df.T.columns[~df.T.isna().all()]]:\n',\
        df.T[df.T.columns[~df.T.isna().all()]],sep='') # para filtrar con .isna().all() por filas
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: dropna (axis=0,how=\'any\',thresh=None,\n \
             subset=None,inplace=False) ______________________________________')
print ('df:\n',df,sep='')
print ('df.dropna(how=\'all\').tail():\n',df.dropna(how='all').tail(),sep='')
print ('df.dropna(how=\'all\',axis=1).tail():\n',\
        df.dropna(how='all',axis=1).tail(),sep='')
print ('df.dropna(subset=[\'serie1\',\'serie2\',\'serie3\']):\n',\
        df.dropna(subset=['serie1','serie2','serie3']),sep='') # subset de col para omitir filas y viceversa
print ('df.dropna(thresh=4):\n',df.dropna(thresh=4)) # thresh = número mínimo de columnas informadas
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: fillna (value=None, method=None, axis=None,\n \
             inplace=False, limit=None, downcast=None) _______________________')
print ('df.fillna (value=9999):\n',df.fillna (value=9999).head(),sep='')
print ('df.fillna(value={\'serie1\':\'SERIE1\',\'serie2\':\'SERIE2\'}):\n',\
        df.fillna(value={'serie1':'SERIE1','serie2':'SERIE2'}).head(),sep='')
print ('df.fillna(value=df.median()):\n',\
        df.fillna(value=df.median()).head(),sep='') # df.median() es un dict.
print ('df.fillna(method=\'ffill\'):\n',df.fillna(method='ffill').head(),sep='') # ffill o forward fill rellena hacia delante (usando el último valor anterior informado)
print ('df.fillna(method=\'bfill\'):\n',df.fillna(method='bfill').head(),sep='') # bfill o backward fill rellena hacia detrás (usando el próximo valor informado). Cuidado con viajes en el tiempo
# df.fillna(methos='ffill').fillna(method='bfill') rellena hacia delante y, donde no hay dato anterior, hacia atrás
print ('df.fillna(value=9999,limit=1):\n',df.fillna(value=9999,limit=1),sep='') # limit con value establece el máximo de valores reemplazados
print ('df.fillna(method=\'ffill\',limit=1):\n',df.fillna(method='ffill',limit=1))
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: eliminar duplicados _______________________________')
L_nombres='bea bea lola lola lola'.split()
L_apellidos=['lópez','lópez','sánchez','sánchez','jalón']
L_valores=[12,12,432,54,23]
df_A=pd.DataFrame({'nombre':L_nombres,'apellido':L_apellidos,'valor':L_valores,})
print ('df_A:\n',df_A,sep='')
print ('df_A.dropduplicates():\n',df_A.drop_duplicates(),sep='') # por defecto: coincidencia en todas las columnas y conserva el primero
print ('df_A.drop_duplicates(subset=[\'nombre\'],keep=\'first\'):\n',\
        df_A.drop_duplicates(subset=['nombre'],keep='first'),sep='') # first por defecto
print ('df_A.drop_duplicates(subset=[\'nombre\',\'apellido\'],keep=\'last\'):\n',\
       df_A.drop_duplicates(subset=['nombre','apellido'],keep='last'),sep='')
print ('df_A.drop_duplicates(subset=[\'nombre\',\'apellido\'],keep=False):\n',\
        df_A.drop_duplicates(subset=['nombre','apellido'],keep=False),sep='') # keep=False elimina todas las ocurrencias duplicadas
print ('df_A.drop_duplicates(subset=[\'nombre\',\'apellido\']).shape[0]:',\
        df_A.drop_duplicates(subset=['nombre','apellido']).shape[0]) # cuenta combinaciones distintas de varias columnas
