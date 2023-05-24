import pandas as pd

df = pd.read_csv('/home/tomhagley/Downloads/trajectories_normalised.csv')

test = df.iloc[4].BodyUpper_x

test = test.replace('\n',' ')
test = test.replace('   ', '  ')
test = test.strip(' []')
test = test.split('  ')