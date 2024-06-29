# Proyecto de Análisis de Hospitalización Post Biopsia Prostática

## Introducción

Este proyecto tiene como objetivo analizar los datos de pacientes sometidos a biopsia prostática en un importante hospital. Se busca identificar las características más importantes de los pacientes que terminan hospitalizados debido a complicaciones infecciosas dentro de los 30 días posteriores al procedimiento. Para ello, se ha desarrollado un modelo predictivo utilizando regresión logística.

## Descripción de los Análisis y Procedimientos

### 1. Exploración y Preparación de los Datos (EDA y ETL)

```python
# importar librerias
# From python
from collections import Counter
# Thirds Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
```

#### Carga de Datos
Los datos fueron proporcionados en un archivo Excel (`BBDD_Hospitalización.xlsx`). La primera tarea fue cargar estos datos en un DataFrame de Pandas para su manipulación y análisis.

```python
# Variable de archivo.
file_path = 'BBDD_Hospitalización.xlsx'

# Carga de datos mediante CSV
datos = pd.read_excel(file_path)
datos.head()
```

![Captura de pantalla 2024-06-28 220705](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/ed9f7495-078a-4f5c-b164-f20ff5c5e9ed)

#### Exploracion  de los datos
> Exploracion de los datos: explorar la estructura del dataset para identificar tipos de datos y valores nulos.
```python
# Detalle de los datos comprendidos en el dataset.
datos.info()
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/ad9c4a98-69d4-4f97-8ed3-a59ff62d7152)

> en el analisis general observamos que contamos con 15 variables categoricas y 5 variables numericas en el data set, asi como tambien multiples valores faltantes en el mismo.

#### Analisis del comportamiento de las variables Categoricas.


```python
# listado de variables categoricas guardada en variable

columnas_categoricas = ['DIABETES','HOSPITALIZACIÓN ULTIMO MES', 'BIOPSIAS PREVIAS', 'VOLUMEN PROSTATICO',
                        'ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS', 'CUP', 'ENF. CRONICA PULMONAR OBSTRUCTIVA', 'BIOPSIA',
                        'NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA', 'FIEBRE','ITU','TIPO DE CULTIVO',
                        'AGENTE AISLADO', 'PATRON DE RESISTENCIA', 'HOSPITALIZACION',
                        ]

# Subplots para visualizar el comportamiento de las variables.
fig, ax = plt.subplots(nrows=len(columnas_categoricas), ncols=1, figsize=(10,45))
fig.subplots_adjust(hspace=0.5)

for i, col in enumerate(columnas_categoricas):
    sns.countplot(x=col, data=datos, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30)
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/dc7a3753-15e9-44b8-bb84-e4fcf72c3a21)

> En este punto podemos determinar que contamos con varias variables categoricas de tipo binarias las cuales nos podrian presentar una facilidad de manejo al convertirlas a numericas, a su vez podemos ver tambien cuales podemos convertir a dummies.
> tambien existen variables con mas de dos clases que pueden normalizarse abinarias y ser convertidas a numericas.

#### Descripcion de variables numericas del dataset.

```python
# Evaluacion de los datos numericos del dataset.
datos.describe()
```

![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/075980d0-c933-41bf-9bf9-a42d498e477a)

> en el siguiente analisis de las variables numericas se pueden observar en edad y PSA, valores atipicos que afectan a la media del los valores.

#### Tratamiento de Valores nulos

```python
# Valores nulos que tenemos en el dataset
datos.isnull().sum()
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/02549980-6954-4094-93bb-46097532b1b4)

## Limpieza de los datos
> Limpiar los datos: se limpiaran los valores nulos y se convertiran los tipos de datos si es necesario.

```python
sns.countplot(x='HOSPITALIZACION', data=datos)
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/e76a312b-8ab0-4888-bb33-559c81913010)
> en el analisis del problema y el requerimiento de la empresa se llego a la decision de tomar la variable Hospital como nuestra Y, por lo que se procede a evaluar de manera mas detallada sus datos, observando que se tiene un desbalance importante en ella.

#### Drop de columnas

```python
# Guardar el nombre de las columbas en una variable
drop_columns = ['EDAD', 'HOSPITALIZACIÓN ULTIMO MES','PSA','BIOPSIAS PREVIAS', 'VOLUMEN PROSTATICO', 
                'NUMERO DE MUESTRAS TOMADAS', 'CUP', 'ENF. CRONICA PULMONAR OBSTRUCTIVA','DIAS HOSPITALIZACION MQ',
                'DIAS HOSPITALIZACIÓN UPC','DIABETES', 'ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS']

# Drop de las columnas y se guarda el resultado en la variable datos.
datos = datos.drop(columns=drop_columns)

datos
```

> Quedarian en total a este punto 8 variables.
> Se realiza la eliminacion de las columnas con baja correlacion en para las predicciones, esta evaluacion se realiza lineas mas abajo, debido a que antes de hacer el analisis de correlacion deben convertirse varias de categoricas a numericas, pero se procede a escribir el paso en este nivel del codigo para que sea mas eficiente al momento de correrse.

#### Drop de filas con espacios Vacios y normalizacion de datos a binarios

```python
# Eliminar filas con datos vacios.
datos = datos.dropna()

# normalizar datos de las filas reemplazando los valores.
datos['NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA'].replace('NO', 0, inplace=True)
```

> Mediante Dropna se procede a eliminar filas con datos vacios  evaluados previamente, a su vez se hace la normalizacion de algunas colunas categoricas en binario.

#### Verificacion de datos vacios o nulos

```python
# Verificar si existe algun dato vacio o nulo en las columnas.
datos.isnull().sum()
```

![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/ec752120-f29c-421c-8731-4b9c3fb95af6)

#### Verificar datos de columna: HOSPITALIZACION

```python
datos[(datos['HOSPITALIZACION']!='NO')&( datos['HOSPITALIZACION']!='SI')]
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/f8e1d12c-cd78-42d0-a354-f752759ad57c)
## Conversion de los datos
> Reemplazar variables categoricas binarias 'Si' y  'NO' con 1 y 0

```python
# Variable para las columnas que se pasaran a True y False.
columnas_binarias = ['FIEBRE', 'ITU', 'HOSPITALIZACION']

for columna in columnas_binarias:
    datos[columna] = datos[columna].map({'SI':1, 'NO':0})

datos.info()
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/ac9a9d49-48fa-4c64-a8be-40adf7df759c)

> Mediante un mapeo con el metodo map() de pandas se procede a reemplazar las clases categoricas de las columnas por datos numerico para poder cargarlos al modelo y se verifica como han quedado los datos.

```python
## Transformar y preparar los datos
> Transformaciones adicionales, como la creación de variables dummie para variables categóricas.

# Variable para las columnas que pasaran a dimmies.
columnas_dummies = ['TIPO DE CULTIVO', 'BIOPSIA', 'AGENTE AISLADO','PATRON DE RESISTENCIA'
                ]

# pasar los datos a tipo dummie pasandole la variable columnas_dummies creada anteriormente.
datos = pd.get_dummies(datos, columns=columnas_dummies,
                        drop_first=True
                        )

datos
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/3ecbe627-717c-4009-b2c6-fab1bd279077)

> Con la funcion Get_dummies() de pandas se procede a convertir las columnas categoricas restantes a dummies para poder trabajar con el modelo escogido.

## Analisis de correlacion

```python
corr = datos.corr()
corr['HOSPITALIZACION'].sort_values(ascending=False)
```

![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/af10bebd-7ec0-4518-92a3-2858f4660ee4)

### Diagrama de correlacion de variables.

```python
# Variable con correlacion de datos desde el Dataset con pandas.
corr = datos.corr()

# Diagaramado con matplotlib y Seaborn
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= datos.columns, 
           yticklabels= datos.columns,
           cmap= 'coolwarm')
plt.show()
```

![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/9ca86b76-46d2-4fa9-885b-bb0477c9f1f7)

> se realiza mediante el metodo corr() de pandas un analisis de correlacion con respecto a la variable HOSPITALIZACION con el fin de evaluar cuales variables darian mejores resultados y a su vez reducir la complejidad del modelo a elaborar (segun esta evaluacion se realiza un drop de dichas columnas unos pasos mas arriba en el codigo).

### Verificacion rapida de que los datos estan sin valores nulos.
```python
datos.isnull().sum()
```

## Implementación del Modelo para la prediccion.
<p>Finalmente, implementamos el modelo de regresión logística utilizando las variables seleccionadas.</p>

## Preparacion de los daros para aplicar los modelos.
> Seleccion, balanceo y Split de datos de entrenamiento y datos de testing

```python
# Seleccionar variables para X e y.
X = datos.drop('HOSPITALIZACION', axis=1)
y = datos['HOSPITALIZACION']
```
> De acuerdo al analisis previo se determino que la variabla a utilizar en 'Y' seria 'HOSPITALIZACION'

```python
# Instancia de Smote
smote = SMOTE()
# se realiza un conteo en la variable y_train para verificar la cantidad de datos al inicio
conteo_i1 = Counter(y)
conteo_i1

#  Se hace el balanceo en los datos para el testeo y el train
X_train, y_train = smote.fit_resample(X,y)

# conteo final de la variable y
conteo_i2 = Counter(y_train)
conteo_i2


print(f'conteo train antes {conteo_i1}')
print(f'conteo train despues {conteo_i2}')
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/35b060ac-8f61-4e23-8bc7-de10ccf5aa49)

> Debido al desbalance en 'Y' se generan datos sinteticos con SMOTE()

```python
# Se selecciona del total de los datos un 20% para el testeo y un 80% para el entrenamiento del modelo. 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size= 0.2, random_state=1990)
```

> Se dividen los datos en X y Y train y X y Y test, para el entrenamiento y el testeo del modelo.

## probar diferentes modelos.

```python
# Variable con las listas de elementos
modelos = []
resultados = []
nombres_modelos = []

# Agregar Modelos a las listas
modelos.append(('Regresion Logistica', LogisticRegression()))
modelos.append(('Arbol de Decision', DecisionTreeClassifier()))
modelos.append(('Bosque de Clasificacion', RandomForestClassifier()))
```

```python
# For para instanciar los modelos con cross_val_score y obtener los parametros de manera individual
for nombre_modelo, model in modelos:
    kfold = StratifiedKFold(n_splits=3,shuffle=True,random_state=1990)
    resultados_cross_value = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    nombres_modelos.append(nombre_modelo)
    resultados.append(resultados_cross_value)

# imprimir resultado de modelos.
for i in range(len(nombres_modelos)):
    print(nombres_modelos[i], resultados[i].mean())
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/369d994d-20b4-4cca-9c4f-39e55759d5d4)

> el mejor resultado en este punto lo arroja el modelo de regresion logistica, por lo que se sera el modelo que se utilice para la predicion.

## Se implementó un modelo de regresión logística para predecir la hospitalización de los pacientes basándose en las características seleccionadas
>Crear Pipeline para agilizar el flujo de trabajo.

```python
# crear Pipeline
modelo = Pipeline((
    ('scale', StandardScaler()), ('log_reg', LogisticRegression(C=10, solver='lbfgs', n_jobs=-1,fit_intercept=True))
))

modelo.fit(X_train,y_train)
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/3ed4733e-55ab-498b-a486-c9975afc1407)

> Prueba de modelo

```python
y_fit = modelo.predict(X_test)
```

> Reporte General del modelo con datos que no conoce.

```python
print(classification_report(y_test, y_fit))
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/b7a4bf96-1672-452b-b1d0-30967871f4cb)

>  Reporte general y Matriz de Confusion
```python
matriz = confusion_matrix(y_test, y_fit)
matriz
```
![image](https://github.com/rammerbot/Project_machine_learning_pacientes_hosp/assets/123994694/16274f7e-68b6-447f-b7fc-811b5b209234)

> Como se puede evaluar el modelo tiene un excelente comportamiento, en la evaluacion de los datos que no conoce ha acertado en todas las prediciones.

### conclusion:

<p>El modelo de regresión logística implementado permitió identificar las características más relevantes asociadas a la hospitalización post biopsia prostática. La precisión del modelo fue evaluada y se presentaron métricas clave como la matriz de confusión y el informe de clasificación. Este análisis proporciona una base sólida para la toma de decisiones en la atención y seguimiento de pacientes sometidos a biopsia prostática tomando en cuenta las siguientes caracteristicas:</p>
<ul>
    <li>BIOPSIA</li>
    <li>NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA </li>
    <li>FIEBRE</li>
    <li>ITU</li>
    <li>TIPO DE CULTIVO</li>
    <li>AGENTE AISLADO</li>
    <li>PATRON DE RESISTENCIA</li>
 </ul>
