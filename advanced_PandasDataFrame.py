import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columnsList=['Nombre','Ciudad']+['variable'+str(i) for i in range(1,4)]
s1=pd.Series(['Ana','Jaén',4,15,27],index=columnsList)
s2=pd.Series(['Lola','Lugo',7,np.nan,23],index=columnsList)
s3=pd.Series(['Bea','Madrid',7,17,29],index=columnsList)
s4=pd.Series(['Pepe','Lugo',8,16,np.nan],index=columnsList)
s5=pd.Series(['Juan','Bilbao',5,12,24],index=columnsList)
s6=pd.Series(['Lola','Bilbao',7,14,25],index=columnsList)
s7=pd.Series(['Juan','Madrid',8,11,23],index=columnsList)
s8=pd.Series(['Ana','Cuenca',3,10,27],index=columnsList)
s9=pd.Series(['Pepe','Lugo',2,19,27],index=columnsList)
df=pd.DataFrame([s1,s2,s3,s4,s5,s6,s7,s8,s9]).set_index('Nombre')
df0=df.copy().reset_index()
print('df:\n',df,sep='')
print ('_____pd.DataFrame: Series combinadas _________________________________')
print ('df0[\'Nombre\']+\'-\'+df0[\'Ciudad\']:\n',\
        df0['Nombre']+'-'+df0['Ciudad'],sep='') # Crea serie a partir de columnas del df
print ('(df0[\'Nombre\']+\'-\'+df0[\'Ciudad\']).nunique():',\
        (df0['Nombre']+'-'+df0['Ciudad']).nunique()) # cuenta pares únicos de las columnas combinadas
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: Agregados por filas y columnas ____________________')
df1=df.copy()
df1.loc['Total_informados por columna']=df.count() # fila de variables informadas por columma
print ('df1.loc[\'Total_informados por columna\']=df.count():\n',df1,sep='')
try:
    df1['Total_informados_por_fila']=df[:-1].count(axis=1) # columna de variables informadas por fila
    print('df1[\'Total_informados_por_fila\']=df[:-1].count(axis=1):\n',df1,sep='')
except: print ('df1[\'Total_informados_por_fila\']=df[:-1].count(axis=1):\n \
                cannot reindex from a duplicate axis') # falla con índices repetidos
df1.reset_index(inplace=True)
df1['Total_informados_por_fila']=df1[:-1].count(axis=1)
print ('df1[\'Total_informados_por_fila\']=df1[:-1].count(axis=1):\n',df1,sep='')
df2=df.copy().reset_index()
df2.loc['Valores_únicos_por columna']=df2.nunique()
print ('df2.loc[\'Valores_únicos_por columna\']=df2.nunique():\n',df2,sep='')
df2['Valores_únicos_por_fila']=df2[:-1].nunique(axis=1)
print('df2[\'Valores_únicos_por_fila\']=df2[:-1].nunique(axis=1):\n',df2,sep='')
df3=df.copy().reset_index()
df3.loc['Totales_por_columnas']=np.sum(df3)
print ('df3.loc[\'Totales_por_columnas\']=np.sum(df3):\n',df3,sep='')
df3['Totales_por_filas']=np.sum(df3,axis=1)
print('df3[\'Totales_por_filas\']=np.sum(df3,axis=1):\n',df3,sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 1 (inner) ___________________________________')
# inner join, solo para las filas que figuran en los dos df (para cada valor común, producto cartesiano de filas en df izda y df dcha)
# outer join, todas las filas, figuren o no en los dos df (respecto a inner, añade filas desparejadas de izda y de dcha)
# left join, el df de la izda con la info extra que aporte sobre sus filas el df de la dcha (respecto a inner, añade filas de izda desparejadas)
# right join, el df de la dcha con la info extra que aporte sobre sus filas el df de la izda (respecto a inner, añade filas de dcha desparejadas)
columnsList_1=['Nombre','Ciudad']+['v'+str(i) for i in range(1,4)]
S1=pd.Series(['Jaime','Madrid']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S2=pd.Series(['Ana','León']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S3=pd.Series(['José','Jaén']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S4=pd.Series(['Lola','Jaén']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S5=pd.Series(['Bea','Lugo']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S6=pd.Series(['Jaime','León']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
S7=pd.Series(['Bea','Madrid']\
            +[i for i in np.random.randint(0,10,3)],index=columnsList_1)
df_1=pd.DataFrame([S1,S2,S3,S4,S5,S6,S7]).set_index('Nombre')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1):\n',df.merge(df_1),sep='') # por defecto, inner join con col. común
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 2 (inner, left_index, right_index) __________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,left_index=True,right_index=True);\n',\
        df.merge(df_1,left_index=True,right_index=True),sep='') # establece join por índices
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 3 (inner, left_on, right_on) ________________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,left_on=\'variable1\',right_on=\'v1\');\n',\
        df.merge(df_1,left_on='variable1',right_on='v1'),sep='') # establece columnas para hacer el join
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 4 (inner, on) _______________________________')
df.rename(columns={'variable1':'variable'},inplace=True)
df_1.rename(columns={'v1':'variable'},inplace=True)
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,on=\'variable\');\n',\
        df.merge(df_1,on='variable'),sep='') # establece columna común para hacer el join
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 5 (inner, suffixes) _________________________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,left_index=True,right_index=True,'
        'suffixes=[\' origen\',\' destino\']));\n',\
        df.merge(df_1,left_index=True,right_index=True,\
        suffixes=[' origen',' destino']),sep='') # establece apellidos para columnas comunes
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 6 (outer) ___________________________________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,how=\'outer\',left_index=True,right_index=True,',
        'suffixes=[\' origen\',\' destino\']));\n',\
        df.merge(df_1,how='outer',left_index=True,right_index=True,\
        suffixes=[' origen',' destino']),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 7 (left) ___________________________________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,how=\'left\',left_index=True,right_index=True,',
        'suffixes=[\' origen\',\' destino\']));\n',\
        df.merge(df_1,how='left',left_index=True,right_index=True,\
        suffixes=[' origen',' destino']),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: merge 8 (right) ___________________________________')
print('df:\n',df,sep='')
print('df_1:\n',df_1,sep='')
print ('df.merge(df_1,how=\'right\',left_index=True,right_index=True,',
        'suffixes=[\' origen\',\' destino\']));\n',\
        df.merge(df_1,how='right',left_index=True,right_index=True,\
        suffixes=[' origen',' destino']),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: variables categóricas (pd.cut) ____________________')
L_nombres='lola bea miguel pepe juan'.split()
L_edades=[12,12,25,36,42]
L_puntuación=[43,22,15,87,9]
df_A=pd.DataFrame({'nombre':L_nombres, 'edad':L_edades,\
                      'puntuación':L_puntuación}).set_index(['nombre'])
print ('df_A:\n',df_A,sep='')
print ('pd.cut(df_A[\'puntuación\'],range(0,100,15)):\n',\
        pd.cut(df_A['puntuación'],range(0,100,15)),sep='')
print ('pd.cut(df_A[\'puntuación\'],10):\n',\
        pd.cut(df_A['puntuación'],10),sep='')
print ('pd.cut(df_A[\'puntuación\'],[0,25,50,75,100]):\n',\
        pd.cut(df_A['puntuación'],[0,25,50,75,100]),sep='')
print ('pd.cut(df_A[\'puntuación\'],[0,25,50,75,100]).value_counts():\n',\
        pd.cut(df_A['puntuación'],[0,25,50,75,100]).value_counts(),sep='')
df_A['I_puntuación']=pd.cut(df_A['puntuación'],\
                     [0,25,50,75,100],labels={'D','C','B','A'}) # columna nueva con edad en intervalos
print ('df_A[\'I_puntuación\']=pd.cut(df_A[\'puntuación\'],'
        '[0,25,50,75,100],labels={\'D\',\'C\',\'B\',\'A\'}):\n',df_A,sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: agregación de datos (.groupby()) __________________')
L_nombresB='miguel lola bea bea lola miguel'.split()
L_edadesB=[12,18,8,47,76,81]
L_puntuaciónB=[25,15,57,74,92,8]
df_B=pd.DataFrame({'nombre':L_nombresB, 'edad':L_edadesB,\
                   'puntuación':L_puntuaciónB}).set_index(['nombre'])
df_A=df_A.append(df_B).sort_index()
df_A['I_puntuación']=pd.cut(df_A['puntuación'],\
                     [0,33,66,100],labels={'baja','media','alta'})
df_A['I_edad']=pd.cut(df_A['edad'],\
               [0,25,50,75,100],labels={'<25','25-50','50-75','>75'})
print ('df_A:\n',df_A,sep='')
print ('df_A.groupby(\'I_edad\').count():\n',
        df_A.groupby('I_edad').count(),sep='')
print ('df_A.groupby(\'I_edad\').mean():\n',
        df_A.groupby('I_edad').mean(),sep='')
print ('df_A.groupby([\'I_edad\',\'I_puntuación\']).count():\n',
        df_A.groupby(['I_edad','I_puntuación']).count(),sep='')
print ('df_A.groupby(\'nombre\')[\'edad\'].mean():\n',
        df_A.groupby('nombre')['edad'].mean(),sep='') # (1)
print ('df_A.groupby(\'nombre\').mean()[\'edad\']:\n',
        df_A.groupby('nombre').mean()['edad'],sep='') # equivalente a (1)
print ('df_A.groupby(\'I_edad\')[\'edad\'].unique():\n',
        df_A.groupby('I_edad')['edad'].unique(),sep='')
print ('df_A.groupby(\'I_edad\')[[\'edad\',\'puntuación\']].agg([\'unique\']):\n',
        df_A.groupby('I_edad')[['edad','puntuación']].agg(['unique']),sep='') # para usar unique como agregador en df: .agg(['unique']) (tratamiento especial)
print ('Iteraciones sobre .groupby():')
for i, group in df_A.groupby('I_edad'): print (i, group.shape)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: .groupby() con varios agregadores _________________')
print ('df_A.groupby(\'nombre\').describe():\n',
        df_A.groupby('nombre').describe(),sep='')
print ('df_A.groupby(\'I_edad\')[\'puntuación\'].agg([\'min\',\'max\']):\n',
        df_A.groupby('I_edad')['puntuación'].agg(['min','max']),sep='') # agg para especiifcar varias funciones como agregadores
print ('df_A.groupby(\'nombre\')[\'edad\'].agg([\'nunique\',\'unique\']):\n',
        df_A.groupby('nombre')['edad'].agg(['nunique','unique']),sep='')
df_lambda=df_A.groupby('I_edad')['puntuación']\
        .agg([lambda serie: serie.max()-serie.min(),\
        lambda serie: serie.min()+(serie.max()-serie.min())/2]) # amplitud y puento medio
print ('df_A.groupby(\'I_edad\')[\'puntuación\']',
        '.agg([lambda serie: serie.max()-serie.min(),',
        'lambda serie: serie.min()+(serie.max()-serie.min())/2]):\n',
        df_lambda,sep='') # amplitud y puento medio
df_N=df_A.groupby('nombre').agg({'edad':['mean','std'],'puntuación':'count'}) # agg con dict style para especificar distintos agregadores para distintas col
print ('df_N=df_A.groupby(\'nombre\')',
        '.agg({\'edad\':[\'mean\',\'std\'],\'puntuación\':\'count\'}):\n',
        df_N,sep='')
print ('df_N[(\'edad\',\'mean\')]:\n',df_N[('edad','mean')],sep='') # recupera serie para una col y un agregador (índice de col múltiple)
print ('df_N[(\'edad\',\'mean\')].sort_values():\n',
        df_N[('edad','mean')].sort_values(),sep='')
print ('df_N.sort_values(by=(\'edad\',\'mean\')):\n',
        df_N.sort_values(by=('edad','mean')),sep='')
print ('df_N[\'edad\'].sort_values(by=\'mean\'):\n',\
        df_N['edad'].sort_values(by='mean'),sep='')
print ('df_N[df_N[\'edad\',\'mean\']>=36]:\n',
        df_N[df_N['edad','mean']>=36],sep='')
df_NP=df_A.groupby('nombre')['puntuación'].agg(['mean','count']) # puntuación media y núm. de puntuaciones por nombre
print ('df_A.groupby(\'nombre\')[\'puntuación\'].agg([\'mean\',\'count\']):\n',
        df_NP,sep='')
print ('df_NP.sort_values(by=\'mean\',ascending=False)[:3]:\n',
        df_NP.sort_values(by='mean',ascending=False)[:3],sep='') # top nombres con mayor puntuación media
print ('df_NP[df_NP[\'count\']>2]',
        '.sort_values(by=\'mean\',ascending=False)[:3]:\n',
        df_NP[df_NP['count']>2].sort_values(by='mean',ascending=False)[:3],
        sep='') # top nombres con mayor puntuación media entre los que tienen al menos 2
df_IP=df_A.groupby('I_puntuación')[['edad']]\
            .apply(lambda df: df['edad'].max()-df['edad'].min()) # apply como agregador (esto no funciona con agg; no permite especificar una serie del df)
print ('df_IP=df_A.groupby(\'I_puntuación\')[[\'edad\']]',
        '.apply(lambda df: df[\'edad\'].max()-df[\'edad\'].min()):\n',
        df_IP,sep='')
df_muestra_estrat_edad=df_A.groupby('I_edad').\
                              apply(lambda serie:serie.sample(len(serie)//2))
print ('df_muestra_estrat_edad=df_A.groupby(\'I_edad\').',
        'apply(lambda serie:serie.sample(len(serie)//2)):\n',
        df_muestra_estrat_edad,sep='')
print ('df_A.groupby(\'I_edad\').count():\n',
        df_A.groupby('I_edad').count(),sep='') # (comprobación de tamaños de muestra)
df_muestra_estrat_edad_sin_group_keys=df_A.groupby('I_edad',group_keys=False).\
                              apply(lambda serie:serie.sample(len(serie)//2))
print ('df_muestra_estrat_edad_sin_group_keys',
        '=df_A.groupby(\'I_edad\',group_keys=False).',
        'apply(lambda serie:serie.sample(len(serie)//2)):\n', # elimina el primer nivel de indices de fila del groupby
        df_muestra_estrat_edad_sin_group_keys,sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: clases de almacenamiento (bucket analysis)\n \
             (cut + groupby __________________________________________________')
rangos_edad=pd.cut(df_A['edad'],[0,20,40,60,80,100],\
                   labels={'e1','e2','e3','e4','e5'})
print ('rangos_edad=pd.cut(df_A[\'edad\'],[0,20,40,60,80,100],'
        'labels={\'e1\',\'e2\',\'e3\',\'e4\',\'e5\'}):\n',
        rangos_edad,sep='')
print ('df_A.groupby(rangos_edad).mean():\n',
        df_A.groupby(rangos_edad).mean(),sep='') # los intervalos para groupby no son columna del df
cuartiles_edad=pd.cut(df_A['edad'],\
                    [df_A['edad'].min(),\
                    df_A['edad'].quantile(q=.25,interpolation='linear'),\
                    df_A['edad'].quantile(q=.5,interpolation='linear'),\
                    df_A['edad'].quantile(q=.75,interpolation='linear'),\
                    df_A['edad'].max()])
print ('cuartiles_edad=pd.cut(df_A[\'edad\'],\n',
                '[df_A[\'edad\'].min(),',
                'df_A[\'edad\'].quantile(q=.25,interpolation=\'linear\'),\n',
                'df_A[\'edad\'].quantile(q=.5,interpolation=\'linear\'),\n',
                'df_A[\'edad\'].quantile(q=.75,interpolation=\'linear\'),\n',
                'df_A[\'edad\'].max()]):\n',
                cuartiles_edad,sep='')
print ('df_A.groupby(cuartiles_edad)[\'edad\'].unique():\n',\
        df_A.groupby(cuartiles_edad)['edad'].unique(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: pivot tables ______________________________________')
df_A['intentos']=[1,2,0,0,3,2,1,4,2,0,1]
df_A['éxito']=[1 if i>=50 else 0 for i in df_A['puntuación'].values]
print ('df_A:\n',df_A,sep='')
PT_0=df_A.pivot_table(index='nombre',columns='edad',values='puntuación')
print ('PT_0=df_A.pivot_table(index=\'nombre\',columns=\'edad\','
                             'values=\'puntuación\'):\n',
                             PT_0,sep='') # aggfunc='mean' por defecto
print ('PT_0>=20:\n',PT_0>=20,sep='')
print ('PT_0[PT_0>=20]:\n',PT_0[PT_0>=20],sep='')
print ('(PT_0>=20).mean():\n',(PT_0>=20).mean(),sep='') # mean sobre una máscara booleana: proporción de trues
print ('PT_0[PT_0>=20].mean():\n',PT_0[PT_0>=20].mean(),sep='')
PT_1=df_A.pivot_table(index='nombre',columns='edad',\
                      values='puntuación',aggfunc='count')
print ('PT_1=df_A.pivot_table(index=\'nombre\',columns=\'edad\',',
                             'values=\'puntuación\',aggfunc=\'count\'):\n',
                             PT_1,sep='')
PT_2=df_A.pivot_table(index='I_edad',columns='I_puntuación',\
                      values='intentos',aggfunc=['mean','count'])
print ('PT_2=df_A.pivot_table(index=\'I_edad\',columns=\'I_puntuación\',',
                        'values=\'intentos\',aggfunc=[\'mean\',\'count\']):\n',
                        PT_1,sep='') # df con 2 niveles de índice de col
for col in PT_2.columns: print ('columnas de PT_2:',col)
print ('PT_2[(\'mean\',\'media\')]:\n',PT_2[('mean','media')],sep='')
PT_3=df_A.pivot_table(index='I_edad',columns=['I_puntuación','éxito'],\
                      values='intentos',aggfunc=['mean','count'])
print ('PT_3=df_A.pivot_table(index=\'I_edad\',',
                             'columns=[\'I_puntuación\',\'éxito\'],',
                             'values=\'intentos\',',
                             'aggfunc=[\'mean\',\'count\']):\n',
                             PT_3,sep='') # df con 3 niveles de índice de col.
for col in PT_3.columns: print ('columnas de PT_3:',col)
PT_4=df_A.pivot_table(index='I_edad',\
                      columns='éxito',\
                      values=['puntuación','intentos'],\
                      aggfunc=['mean','count'])
print ('PT_4=df_A.pivot_table(index=\'I_edad\',',
                      'columns=\'éxito\',',
                      'values=[\'puntuación\',\'intentos\'],',
                      'aggfunc=[\'mean\',\'count\'])',
                      PT_4,sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.DataFrame: modificar valores con .where() y .mask() __________')
print ('df_A[\'éxito\'].where(df_A[\'éxito\']==1):\n',\
        df_A['éxito'].where(df_A['éxito']==1).head(),sep='') # por defecto, other=nan
print ('df_A[\'éxito\'].where(df_A[\'éxito\']==1,\'fail\'):\n',\
        df_A['éxito'].where(df_A['éxito']==1,'fail').head(),sep='') # where reemplaza los valores que no cumplen la condición
print ('df_A[\'éxito\'].where(df_A[\'éxito\']==1,other=\'fail\'):\n',\
        df_A['éxito'].where(df_A['éxito']==1,other='fail').head(),sep='') # equivalente (con other= explícito)
df_C=df_A.drop(['I_puntuación','I_edad'],axis=1)
print ('df_C:\n',df_C.head(),sep='')
print ('df_C.where(df_C>10,-df_C):\n',df_C.where(df_C>10,-df_C).head(),sep='')
par=df_C%2==0
print ('par=df_C%2==0:\n',par.head(), sep='')
print ('df_C.where(par,\'impar\'):\n',df_C.where(par,'impar').head(),sep='')
print ('df_C.where(par,-df_C)==np.where(par,df_C,-df_C):\n',\
        (df_C.where(par,-df_C)==np.where(par,df_C,-df_C)).head(),sep='') # df1.where(cond,df2)=np.where(cond,df1,df2)
print ('df_C.where(par,-df_C)==df_C.mask(~par,-df_C):\n',\
        (df_C.where(par,-df_C)==df_C.mask(~par,-df_C)).head(),sep='') # mask reemplaza los valores que sí cumplen la condición
print ('df_C[\'éxito\'].where(df_C[\'éxito\']==0,\'pass\')',
        '.mask(df_C[\'éxito\']==0,\'fail\'):\n',
        df_C['éxito'].where(df_C['éxito']==0,'pass')\
        .mask(df_C['éxito']==0,'fail').head(),sep='')
print ('df_C[\'éxito\'].where(df_C[\'éxito\']==0,\'pass\')',
        '.mask(df_C[\'éxito\']==0,\'fail\')==',
        'f_A[\'éxito\'].where(df_A[\'éxito\']==1,other=\'fail\')',
        '.where(df_A[\'éxito\']==0,other=\'pass\'):\n',
        (df_C['éxito'].where(df_C['éxito']==0,'pass')\
        .mask(df_C['éxito']==0,'fail')==\
        df_A['éxito'].where(df_A['éxito']==1,other='fail')\
        .where(df_A['éxito']==0,other='pass')).head(),sep='')
df_A['pass_fail']=np.where(df_A['éxito']==1,'pass','fail')
print ('df_A[\'pass_fail\']=np.where(df_A[\'éxito\']==1,\'pass\',\'fail\'):\n',\
        df_A,sep='')
