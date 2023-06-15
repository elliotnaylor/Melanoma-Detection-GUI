import bnlearn as bn
import numpy as np
import pandas as pd
import matplotlib as plot

class bayesianFusion:

    def __init__(self, path):
        #1. Importing data
        csv = np.genfromtxt (path, delimiter=",")

        sampleData = ['Diagnosis', 'Asymmetry', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks']

        self.df = pd.DataFrame(csv, columns = sampleData)

        #2. Making directed acycle graph (DAG)
        #edges = [('Globules','Diagnosis'), ('Milia','Diagnosis'), ('Negative','Diagnosis'), ('Pigment','Diagnosis'), ('Streaks','Diagnosis'), ('Structures', 'Diagnosis')]

        edges = [
            ('Asymmetry', 'Diagnosis'),
            ('Globules','Diagnosis'), 
            ('Milia','Diagnosis'), 
            ('Negative','Diagnosis'), 
            ('Pigment','Diagnosis'), 
            ('Streaks','Diagnosis')]

        DAG = bn.make_DAG(edges, methodtype='bayes')

        #fig = bn.plot(DAG)

        #3. Train model
        self.DAG_update = bn.parameter_learning.fit(DAG, self.df)

        #Generate samples and re-train based on those samples
        df_sampling = bn.sampling(DAG, n=10000)
        self.DAG_update = bn.parameter_learning.fit(DAG, df_sampling)

        #Pout = bn.predict(self.DAG_update, self.df, variables=['Diagnosis', 'Globules', 'Milia', 'Negative', 'Pigment', 'Streaks', 'Structures'])

        #print(Pout)

    def predict(self, _a, _g, _m, _n, _p, _s):

        #4. Predict
        query = bn.inference.fit(self.DAG_update,  variables=['Diagnosis'], evidence={ 'Asymmetry': _a, 'Globules':_g, 'Milia':_m, 'Negative':_n, 'Pigment':_p, 'Streaks':_s})

        #print(query)
        #print(bn.query2df(query))
        

        return query.values
    
        


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