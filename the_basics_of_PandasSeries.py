import pandas as pd
import numpy as np
#help (pd.Series)
print ('_____pd.Series: establecer índices ___________________________________')
Animals=pd.Series(['Tiger','Bear', None, np.nan]) # Series a partir de lista. Id automático
print ('Animals:\n',Animals,sep='')
Animals.index=list('abcd') # Sustituye índices por a,b,c,d (lista)
print ('Animals:\n',Animals,sep='')
Animals.index=['bicho'+str(i) for i in range(1,5)] # Sustitye índices (en lista)
print ('Animals:\n',Animals,sep='')
print ('_____pd.Series. componentes: listas de índices, valores e items ______')
print ('Animals.index:',Animals.index) # Lista de índices de la serie
print ('Animals.values:',Animals.values) # Lista de valores de la serie
print ('Animals.items():',Animals.items()) # Lista de pares (índice, valor)
for tupla in Animals.items(): print (tupla) # tuplas en .items()
for i,v in Animals.items(): print (i,v) # índices y valores en .items()
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series. recuperar (.iloc[],.loc[],[]) ________________________')
print ('Animals.iloc[1]:',Animals.iloc[1]) # .iloc[i] para recuperar valor por posición
print ('Animals.iloc[[1,2]]:\n',Animals.iloc[[1,2]],sep='') # .iloc[[,]] para recuperar varios valores por posición
print ('Animals.iloc[1:4]:\n',Animals.iloc[1:4],sep='') # .iloc[:] para recuperar valores por rango de posición
print('Animals[1]:',Animals[1]) # [] Por posición. No recomendable (pueden confundirse índices automáticos y no automáticos)
print('Animals[[1,3]]:\n',Animals[[1,3]],sep='') # [[,]] Varios valores por posición. No recomendable
print('Animals[1:4]:\n',Animals[1:4],sep='') # [:] Por rango de posición. No recomendable
print ('Animals.loc[\'bicho1\']:',Animals.loc['bicho1']) # .loc['index'] para recuperar valores por índice
print ('Animals.loc[[\'bicho1\',\'bicho3\']]:\n',\
        Animals.loc[['bicho1','bicho3']],sep='') # .iloc[[,]] para recuperar varios valores por índice
# Cuidado: .loc[:] (con rangos) produce aberraciones
print ('Animals[\'bicho1\']:',Animals['bicho1']) # [] Recupera valores por índice
print ('Animals[[\'bicho2\',\'bicho4\']]:\n',\
        Animals[['bicho2','bicho4']],sep='') # [[,]] recupera varios valores por índice
print ('Animals[\'bicho2\':\'bicho4\']:\n',Animals['bicho2':'bicho4'],sep='') # [:] recupera valores por rango
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series. añadir y sobreescribir (.iloc[],.loc[],[]) ___________')
#Animals.iloc[5]='tortuga' # Falla (fuera de rango): no sirve para añadir valores
Animals.iloc[2]='ballena' # .iloc[] para sobreescribir valor por posición
print ('Animals.iloc[2]=\'ballena\':\n',Animals,sep='')
Animals.loc['bicho5']='pato' # .loc[] para añadir valor nuevo por índice
print ('Animals.loc[\'bicho5\']=\'pato\':\n',Animals,sep='')
Animals.loc['bicho5']='perezoso' # .loc[] para sobreescribir valor por índice
print ('Animals.loc[\'bicho5\']=\'perezoso\':\n',Animals,sep='')
Animals['bicho6']='avestruz' # [] para añadir o sobreescribir por índice
print ('Animals[\'bicho6\']=\'avestruz\':\n', Animals,sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series. drop y truncate ______________________________________')
print ('Animals.drop(\'bicho5\'):\n',Animals.drop('bicho5'),sep='') # Elimina por índice. Admite inplace=True
for i,v in Animals.items(): #bucle para elminar por valor
    if v=='Tiger': print(Animals.drop(i)) # inplace=True para conservar cambios
A=pd.Series(np.linspace(1,22,8),list('abcdefgh')) # Serie a partir de array de numpy
print ('A:\n',A,sep='')
print ('A.truncate(before=\'c\',after=\'f\'):\n',\
        A.truncate(before='c',after='f'),sep='') # Trunca por índice
print ('A[2:6]:\n',A[2:6],sep='') #  (slice requiere conocer la posición)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series: head, tail, len, size y shape) _______________________')
B=pd.Series(np.random.randint(0,9,100)) # serie a partir de lista aleatoria
print ('B.head():\n',B.head(),sep='') # presenta los 5 primeros elementos
print ('B.head(7):\n',B.head(7),sep='') # presenta los n primeros elementos
print ('B.tail():\n',B.tail(),sep='') # presenta los 5 últimos elementos
print ('B.tail(7):\n',B.tail(7),sep='') # presenta los n últimos elementos
print ('B.sample():\n',B.sample(),sep='') # presenta un elemento elegido aleatoriamente
print ('B.sample(5):\n',B.sample(5),sep='') # presenta n elementos elegidos aleatoriamente
print ('len(B):',len(B)) # longitud de la serie
print ('B.size:',B.size) # =filas x columnas (para Series, igual que len)
print ('B.shape:',B.shape) # Orden (filas,columnas -en series aparece vacío-)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.nan y None ___________________________________________________')
Numbers=pd.Series([1,2,None,np.nan])
print ('Numbers:\n',Numbers,sep='')
print ('Numbers[Numbers>1]:\n',Numbers[Numbers>1],sep='')
print ('Numbers[Numbers != np.nan]:\n',Numbers[Numbers != np.nan],sep='')
print ('np.nan==None:',np.nan==None) # (False)
print ('np.nan==np.nan:',np.nan==np.nan) # (False)
print ('np.isnan(np.nan):',np.isnan(np.nan)) # (True) np.isnan es una función especial
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series a partir de diccionarios ______________________________')
D={'bea':'lópez','fer':'garcía','elena':'sánchez'}
print ('D:',D)
Names=pd.Series(D) # Series a partir de dict. Id= key
print ('Names:\n',Names,sep='')
print ('Names.index',Names.index)
print ('_____pd.Series. Escoger elementos de un diccionario __________________')
nombres={'gato':'miau','perro':'guau','pez':'glu'}
Nombres=pd.Series(nombres,index=['perro', 'gato', 'tortuga']) # Serie como subserie de dict (con índices descuadrados)
print ('Nombres:\n',Nombres,sep='')
print ('Nombres.index:', Nombres.index)
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series. índices repetidos ____________________________________')
Q=pd.Series(['uno','dos','tres','cuatro'],index=['one','one','three','four']) # índices no únicos
print ('Q:\n',Q,sep='')
print ('Q.index:',Q.index)
print('Q.loc[´one´]:\n',Q.loc['one'],sep='') # el valor asoc. a un índice repetido es una subserie
print('Q.loc[´one´][1]:',Q.loc['one'][1])
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print('_____pd.Series. Concatenar series con .append _________________________')
Nnombres=Names.append(Nombres)
print ('Names.append(Nombres):\n',Nnombres,sep='') # append para unir series (serie nueva)
print('Names:\n',Names,sep='') # no cambia la serie original
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____numpy con pandas (broadcasting): estadística descriptiva ________')
A=pd.Series([9,16,9,4,1])
print ('A:\n',A,sep='')
print ('np.sum(A):',np.sum(A))
print ('A.sum():',A.sum())
print ('A.count():', A.count()) # no. de elementos
print ('A.unique():',A.unique())
print('A.nunique():',A.nunique())
print('A.value_counts():\n',A.value_counts(),sep='')
print ('A.mean():', A.mean())
print ('A.median():', A.median())
print ('A.quantile(q=0.25,interpolation=\'linear\'):',\
        A.quantile(q=0.25, interpolation='linear')) # más opciones de interpolación
print ('A.min():', A.min())
print ('A.max():', A.max())
print ('A.std():', A.std())
print ('A.describe():\n',A.describe(),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____numpy con pandas: operaciones con series  _______________________')
AA=A*2+1
print('AA:\n',AA,sep='')
B=pd.Series(np.random.randint(0,9,100)) # Serie de valores aleatorios
C=A+B
print('C:\n',C.head(7),sep='')
import math
print ('A.apply(math.sqrt):\n',A.apply(math.sqrt),sep='')
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series: sort, rank, unique, count, is in _____________________')
A.index=list('vgwkr')
print ('A:\n',A,sep='')
print ('A.sort_values():\n',A.sort_values(),sep='')
print ('A.sort_index():\n',A.sort_index(),sep='')
print ('A.rank():\n',A.rank(),sep='') # Posición de cada valor. Resuelve empates con promedio
try: print('A[A.rank()==2]:',A[A.rank()==2]) # Valor en cierta posición (ver máscaras booleanas)
except: pass # try/except porque la máscara falla si, por promedio, no existe la posición
print ('A.rank(method=\'dense\'):\n',A.rank(method='dense'),sep='') # distinto output que en Jupyter Notebook
print ('A.rank(ascending=False):\n',A.rank(ascending=False),sep='')
print ('A.rank(ascending=False,pct=True):\n',\
        A.rank(ascending=False,pct=True),sep='')
print ('A.rank(ascending=False,pct=True)<0.7:\n',\
        A.rank(ascending=False,pct=True)<0.7,sep='') # máscara booleana
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series: unique, nunique, value_counts, is in _________________')
print ('A.unique():',A.unique(),sep='') # Lista de valores únicos
print ('A.unique():',A.nunique(),sep='') # número de valores únicos
print('A.value_counts():\n',A.value_counts(),sep='') # Valores y frecuencias
print ('A.isin([3,5,9])):\n', A.isin([3,5,9]),sep='') # Para cada valor, T/F según esté o no en lista
print ('A[A.isin([3,5,9])]:\n',A[A.isin([3,5,9])],sep='') # .isin() usado como máscara
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series: máscaras booleanas (filtros) _________________________')
print ('A:\n',A,sep='')
print ('A>=5:\n',A>=5,sep='') # T/F según cada valor cumpla o no la condición
print ('A[A>=5]:\n',A[A>=5],sep='') # Solo elementos de la serie que cumplen la condición
print ('(A[A>=5) & (A%2==0)]:\n',A[(A>=5) & (A%2==0)],sep='') # & es y: >=5 y par
print ('A[(A>=5) | (A%2==0)]:\n',A[(A>=5) | (A%2==0)],sep='') # | es o: >=5 o par
print ('A[~(A%2==0)]:\n',A[~(A%2==0)],sep='') # ~ es no: solo impares
next=input('¿Seguir? q para salir')
if 'q' in list(next): quit()
print ('_____pd.Series: apply ________________________________________________')
# apply sobre pd.Series o sobre pd.DataFrame. Puede ser element-wise o un agregado dependiendo del contexto
# applymap sobre cada elemento (celda) de un pd.DataFrame
# map sobre cada elemento de una pd.Series (a menudo, equivalente a apply)
L=[1,5,4,9,16,25,27,36]
s=pd.Series(L,index=['v'+str(i) for i in range(1,len(L)+1)])
print ('s:\n',s,sep='')
print ('np.sqrt(s):\n',np.sqrt(s),sep='') # ejemplo de funcion que se puede aplicar sobre pd.series
import math
try: print (math.sqrt(s))
except: print ('No es posible aplicar directamente math.sqrt a una serie')
print ('s.apply(math.sqrt):\n',s.apply(math.sqrt),sep='') # apply para aplicar funciones y métodos sobre pd.Series y df. No permanente
# apply, applymap y map no son métodos vectorizados: mucho más lentos que las funciones incorporads en Pandas
s_par=s.apply(lambda x: 'par' if x%2==0 else 'impar') # sobre la serie (devuelve índices)
print ('s.apply(lambda x: \'par\' if x%2==0 else \'impar\'):\n',s_par,sep='')
s_par=s.map(lambda x: 'par' if x%2==0 else 'impar') # sobre la serie (devuelve índices)
print ('s.map(lambda x: \'par\' if x%2==0 else \'impar\'):\n',s_par,sep='')
es_cuadrado=lambda x: \
            'cuadrado' if math.modf(math.sqrt(x))[1]==math.sqrt(x) \
            else 'no cuadrado' # math.modf(x) = (x-E(x), E(x)) (parte decimal y parte entera)
s_cuadrado=s.apply(es_cuadrado) # sobre la serie (devuelve índices)
print ('s_cuadrado=s.apply(es_cuadrado):\n',s_cuadrado,sep='')
entera_y_decimal=lambda x: (int(x),x-int(x)) # función lambda para parte entera y decimal sin math
print ('entera_y_decimal(7.4):',entera_y_decimal(7.4))
print('s.apply(math.sqrt).apply(entera_y_decimal):\n',\
       s.apply(math.sqrt).apply(entera_y_decimal),sep='')
print('s.apply(math.sqrt).apply(entera_y_decimal).loc[\'v7\'][0]:',\
     s.apply(math.sqrt).apply(entera_y_decimal).loc['v7'][0])
tupla_0=lambda x:x[0] # para separar la parte entera: aqui (abajo) map y apply son intercambiables
print ('s.apply(math.sqrt).apply(entera_y_decimal).apply(tupla_0):\n',\
      s.apply(math.sqrt).apply(entera_y_decimal).apply(tupla_0),sep='')
print ('s.map(math.sqrt).map(entera_y_decimal).map(tupla_0):\n',\
      s.map(math.sqrt).map(entera_y_decimal).map(tupla_0),sep='')
print ('s.apply(print):')
s.apply(print) # elemento a elemento (sin índices)
print ('s.map(print):')
s.map(print) # equivalente a s.apply(print)
print ('s:\n',s,sep='')
