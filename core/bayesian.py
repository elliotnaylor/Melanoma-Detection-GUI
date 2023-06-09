import bnlearn as bn
import numpy as np
import pandas as pd
import matplotlib as plot

#1. Importing data
path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth_complete3.csv'

csv = np.genfromtxt (path, delimiter=",", )

sampleData = ['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks']

df = pd.DataFrame(csv, columns = ['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks'])

#2. Making directed acycle graph (DAG)
edges = [('Globules','Diagnosis'), ('Milia','Diagnosis'), ('Negative','Diagnosis'), ('Pigment','Diagnosis'), ('Streaks','Diagnosis')]

DAG = bn.make_DAG(edges, methodtype='bayes')

fig = bn.plot(DAG)

#3. Train model
DAG_update = bn.parameter_learning.fit(DAG, df)

#Generate samples and re-train based on those samples
df_sampling = bn.sampling(DAG, n=10000)
DAG_update = bn.parameter_learning.fit(DAG, df_sampling)

#4. Predict
query = bn.inference.fit(DAG_update,  variables=['Diagnosis'], evidence={'Globules':0, 'Milia':0, 'Negative':0, 'Pigment':0, 'Streaks':1})

print(query)
print(bn.query2df(query))

Pout = bn.predict(DAG_update, df, variables=['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks'])

print(Pout)


"""
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

path = 'D:/Datasets/ISIC_2018/ISIC_2017_GroundTruth_complete3.csv'
path_pred = 'D:/Datasets/ISIC_2018/ISIC_2017_Pred.csv'

train_file = np.genfromtxt (path, delimiter=",", dtype=str)
pred_file = np.genfromtxt (path_pred, delimiter=",", dtype=str)

train_file = train_file.tolist()

train = pd.DataFrame(train_file, columns = ['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks'])
#pred = pd.DataFrame(pred_file, columns = ['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks'])

model = BayesianNetwork([('Globules','Diagnosis'), ('Milia','Diagnosis'), ('Negative','Diagnosis'), ('Pigment','Diagnosis'), ('Streaks','Diagnosis')])

model.fit(train)

#pred.drop('Diagnosis', 1, inplace=True)
#y_pred = model.predict(pred)
#y_pred

infer = VariableElimination(model)
g_dist = infer.query(variables=['Diagnosis'], evidence={'Globules' : '1', 'Milia' : '1', 'Negative' : '1', 'Pigment' : '1', 'Streaks' : '1'})

print(g_dist)
"""