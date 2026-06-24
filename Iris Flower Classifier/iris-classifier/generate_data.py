from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = [iris.target_names[t] for t in iris.target]
df.to_csv('/home/claude/iris-classifier/data/IRIS.csv', index=False)
print("Dataset saved. Shape:", df.shape)
print(df.head())
