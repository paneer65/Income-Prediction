import numpy as np
import pandas
from sklearn.impute import SimpleImputer as Imputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split



def cleaning(df):
 #checking for null colums
 print (df.isnull().sum())

 #Replacing different formats of missing values using simpleImputer
 imputer = Imputer(missing_values = np.nan, strategy = 'median')
 imputer = imputer.fit(df[['Year of Record']])
 df[["Year of Record"]] = imputer.fit_transform(df[["Year of Record"]]).ravel()
 imputer = imputer.fit(df[['Age']])
 df["Age"] = imputer.fit_transform(df[["Age"]]).ravel()
 imputer = Imputer(missing_values = np.nan, strategy = 'most_frequent')
 imputer = imputer.fit(df[['Gender']])
 df["Gender"] = imputer.fit_transform(df[["Gender"]]).ravel()

 imputer = Imputer(missing_values = '0', strategy = 'most_frequent')
 imputer = imputer.fit(df[['Gender']])
 df["Gender"] = imputer.fit_transform(df[["Gender"]]).ravel()

 imputer = Imputer(missing_values = 'Unknown', strategy = 'most_frequent')
 imputer = imputer.fit(df[['Gender']])
 df["Gender"] = imputer.fit_transform(df[["Gender"]]).ravel()

 imputer = Imputer(missing_values = 'unknown', strategy = 'most_frequent')
 imputer = imputer.fit(df[['Gender']])
 df["Gender"] = imputer.fit_transform(df[["Gender"]]).ravel()

 imputer = Imputer(missing_values = np.nan, strategy = 'most_frequent')
 imputer = imputer.fit(df[['Profession']])
 df["Profession"] = imputer.fit_transform(df[["Profession"]]).ravel()

 imputer = Imputer(missing_values = np.nan, strategy = 'most_frequent')
 imputer = imputer.fit(df[['University Degree']])
 df["University Degree"] = imputer.fit_transform(df[["University Degree"]]).ravel()
 imputer = Imputer(missing_values = '0', strategy = 'most_frequent')
 imputer = imputer.fit(df[['University Degree']])
 df["University Degree"] = imputer.fit_transform(df[["University Degree"]]).ravel()

 imputer = Imputer(missing_values = np.nan, strategy = 'most_frequent')
 imputer = imputer.fit(df[['Hair Color']])
 df["Hair Color"] = imputer.fit_transform(df[["Hair Color"]]).ravel()

 imputer = Imputer(missing_values = '0', strategy = 'most_frequent')
 imputer = imputer.fit(df[['Hair Color']])
 df["Hair Color"] = imputer.fit_transform(df[["Hair Color"]]).ravel()

 imputer = Imputer(missing_values = 'Unknown', strategy = 'most_frequent')
 imputer = imputer.fit(df[['Hair Color']])
 df["Hair Color"] = imputer.fit_transform(df[["Hair Color"]]).ravel()

 return df


def main():
 #Loading training data
 df = pandas.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
 #cleaning the data and storing it into new file train.csv
 clean = cleaning(df)
 clean.to_csv('train.csv')
 #Loading test data
 df = pandas.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
 del df['Income']  #target column is empty
 #cleaning the data and storing it into new file train.csv
 clean = cleaning(df)
 clean.to_csv('test.csv')

 #loading clean train and test dataset
 df = pandas.read_csv('train.csv')
 tdf = pandas.read_csv('test.csv')
 #poltting correlation matrix for train data
 plt.figure(figsize=(12,10))
 cor = df.corr()
 sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
 plt.show() #Year of record and age most relevant
 #separate target variable from traing set
 y = df['Income in EUR'].values

 #through trial and error selected feautures that perfom the best together
 del df['Income in EUR']
 del df['Instance']
 ins = tdf['Instance'].values
 del tdf['Instance']
 #del df['Year of Record']
 #del tdf['Year of Record']
 del df['Gender']
 del tdf['Gender']
 #del df['Age']
 #del df['Age']
 #del df['Country']
 #del tdf['Country']
 del df['Size of City']
 del tdf['Size of City']
 #del df['Profession']
 #del tdf['Profession']
 del df['University Degree']
 del tdf['University Degree']
 del df['Wears Glasses']
 del tdf['Wears Glasses']
 del df['Hair Color']
 del tdf['Hair Color']
 #del df['Body Height [cm]']
 #del tdf['Body Height [cm]']

 #calculting no.of rows in train dataset
 a = df.shape[0]
 #merging train and test data to apply same encoding
 all = pandas.concat([df, tdf], axis=0)
 #Using dictVectorizer to encode everything
 all = all.to_dict(orient='records');
 dv_X = DictVectorizer(sparse=False)
 all = dv_X.fit_transform(all)
 #Separating dataset to get back training and test dataset after encoding
 train = all[:a]
 test = all[a:]
 #spliting training dataset; preparing data for model
 xtrain, xtest, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)
 #Initialising linear Model; tried several model to find out Bayesian Ridge works slightly better
 regressor = linear_model.BayesianRidge ()
 #training the model
 regressor.fit(xtrain, y_train)

 #testing the model
 y_pred = regressor.predict(xtest)
 print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
 print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
 #applying model on testdataset
 y_pred = regressor.predict(test)
 #saving results to submission csv file
 dataset = pandas.DataFrame({'Instance': ins, 'Income': y_pred}, columns=['Instance', 'Income'])
 dataset.to_csv('results.csv')










if __name__ == '__main__':
  main()
