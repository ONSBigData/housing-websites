__author__ = 'ONS-BIG-DATA'

# Use scikitlearn to look at machine learning techniques
# for identifying park homes from Zoopla data
# Karen Gask
# 31/07/15

# Import packages
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from scipy.stats import sem

# Import all homes data collected from Zoopla API
# This is for PO, SO, NR and TR postcode areas
homes = pd.read_csv('C:/Users/ONS-BIG-DATA/Documents/Zoopla/park_homes_logistic2.csv')

print(homes.head())
print(homes.dtypes)

####### Data wrangling

# Create a function which z-scores for rental and sale prices independently
# Adds a new variable, zprice, from price
# (X(i) - X(mean)) / (X(StDev))
def zscores(df):
    # Sale
    df['saleprice'] = np.where(df['listing_status']=='sale', df['price'], 0)
    df['zsaleprice'] = np.where(df['listing_status']=='sale', (df['saleprice'] - df['saleprice'].mean()) / (df['saleprice'].std()), 0)
    # Rent
    df['rentprice'] = np.where(df['listing_status']=='rent', df['price'], 0)
    df['zrentprice'] = np.where(df['listing_status']=='rent', (df['rentprice'] - df['rentprice'].mean()) / (df['rentprice'].std()), 0)
    # Combine both into one z-scored price variable
    df['zprice'] = df[['zsaleprice','zrentprice']].sum(axis=1)
    df = df.drop(['saleprice','rentprice','zsaleprice','zrentprice'], axis=1)
    return df

zscores(homes)

# Need to see whether certain phrases are in the property description
def add_caravan_booleans(df):
    df['description_lower'] = df['description'].str.lower()
    df['holiday_park'] = np.where(df['description_lower'].str.contains("holiday park"), 1, 0)
    df['holiday_village'] = np.where(df['description_lower'].str.contains("holiday village"), 1, 0)
    df['chalet_text'] = np.where(df['description_lower'].str.contains("chalet"), 1, 0)
    df['park_home'] = np.where(df['description_lower'].str.contains("park home"), 1, 0)
    df['static_home'] = np.where(df['description_lower'].str.contains("static home"), 1, 0)
    df['mobile_home'] = np.where(df['description_lower'].str.contains("mobile home"), 1, 0)
    df['park_lodge'] = np.where(df['description_lower'].str.contains("park lodge"), 1, 0)
    df['static_caravan'] = np.where(df['description_lower'].str.contains("static caravan"), 1, 0)
    return df

add_caravan_booleans(homes)

# Property type has lots of different values (eg. detached house, semi-detached house etc.)
# Explore number of cases per property type
print(homes.groupby(by=['property_type'])['listing_id'].count())
# I can narrow these down in feature selection later

# Need to one hot encode property_type as this needs to be a numeric value for scikit learn
homes_prop_type = pd.get_dummies(homes['property_type'])
homes = pd.concat([homes, homes_prop_type], axis=1)

# Convert listing_status (whether for sale or for rent to a numeric value, 1=sale, 0=rent)
homes['listing_status'] = np.where(homes['listing_status']=='sale',1,0)

# Scikit learn needs a numpy array, not a pandas dataframe
# Keep only relevant variables which may be able to predict whether you have a park home

# x = explantory features
x = homes[['listing_status','num_bedrooms','zprice','holiday_park','holiday_village','chalet_text','park_home','static_home',
           'mobile_home','park_lodge','static_caravan','Barn conversion','Bungalow','Chalet','Cottage',
           'Country house','Detached bungalow','Detached house','End terrace house','Equestrian property','Farm','Farmhouse',
           'Flat','Land','Link-detached house','Lodge','Maisonette','Mews house','Mobile/park home','Parking/garage',
           'Semi-detached bungalow','Semi-detached house','Studio','Terraced bungalow','Terraced house','Town house','Villa']].astype(np.float64)
X = np.array(x)

# y = label or target variable which you are trying to predict. Must be a number
y = homes[['parkhome']]
y = np.array(y)
y = y.ravel()
print(sum(y), "cases labelled as park homes out of", len(y), "(%0.2f per cent)" % (sum(y)/len(y)*100))

# Print information about x and y
print(X[0], y[0])
print(X.shape, y.shape)
print(type(X), type(y))

####### Principal components analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)

# Plot first two eigenvectors with different colours for y (whether caravan home or not)
pca = PCA(n_components=2).fit(X)
X_reduced = pca.transform(X)
import matplotlib.pyplot as plt
cmap = 'jet'
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap=cmap)
plt.show()

####### Machine learning

# Train / Test split here
# Train to decide upon the best type of model eg. logistic regression, SVM etc.
# Cross validation to a final evaluation of the goodness-of-fit of the chosen model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Check out percentage of park homes in training and testing datasets
def positive_class(y):
    print("%0.2f per cent" % (sum(y)/len(y)*100), "of cases are park homes")

positive_class(y_train)
positive_class(y_test)

# Create a K-fold cross validation iterator with F1 score instead of accuracy
# K is set to 10 - change if need be
def fit_evaluate_cv(model, X, y):
    model.fit(X, y)
    cv = StratifiedKFold(y, 10, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(scores)
    print("Mean F1 score: %0.3f (+/- %0.3f)" % (np.mean(scores), sem(scores)))
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    print("Mean precision: %0.3f" % (np.mean(precision)))
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    print("Mean recall: %0.3f" % (np.mean(recall)))

# Feature selection with 10-fold cross validation
from sklearn.feature_selection import RFECV
def select_features(model, X, y):
    feature_names = list(x.columns.values)
    rfecv = RFECV(model, step=1, cv=10)
    rfecv = rfecv.fit(X, y)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), feature_names)))

# Run model with features selected in select_features function
def fit_evaluate_cv_selected_features(model, X, y):
    rfecv = RFECV(model, step=1, cv=5, scoring='f1')
    rfecv = rfecv.fit(X, y)
    X_transformed = rfecv.transform(X)
    print("Shape of full training set: ",X.shape)
    print("Shape of reduced training set: ", X_transformed.shape)
    fit_evaluate_cv(model, X_transformed, y)

# Logistic regression with all features
# L1 regularisation better than L2 if you have less observations than features
# In my case I have many more observations than features so L2 thought to be better
# L2 also performs slightly better here
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(class_weight='auto', penalty='l2') # class_weight is for unbalanced datasets, L2 regularisation
fit_evaluate_cv(log, X_train, y_train)
# Select features to use
select_features(log, X_train, y_train)
# Logistic regression with reduced number of features
fit_evaluate_cv_selected_features(log, X_train, y_train)
# So reducing number of features from 37 to 3 increases mean F1 score from 0.828 to 0.867

# Decision tree
# Entropy better than Gini here - gini F1=0.494, entropy F1=0.862
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=9, class_weight="auto")
# Use with all features
fit_evaluate_cv(dtree, X_train, y_train)
# Can also do feature_importances_ here

# Random forests
# Entropy better than Gini here - gini F1=0.895, entropy F1=0.900
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy')
fit_evaluate_cv(rf, X_train, y_train)
feature_names = list(x.columns.values)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True))

# Support Vector Machines grid search for optimisation
# cv: If cv is an integer input, if y is binary, StratifiedKFold used, so StratifiedKFold is used here
# Also used class_weight='auto' to add weight to unbalanced y_train
from sklearn import svm, grid_search
import time
start_time = time.time()
param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1.0], 'kernel': ['linear','rbf']} ]
svmsearch = svm.SVC(class_weight='auto')
svmmodel = grid_search.GridSearchCV(svmsearch, param_grid, cv=3, scoring='f1')
svmmodel.fit(X_train, y_train)
print(svmmodel.best_params_)
print("--- %s seconds ---" % (time.time() - start_time))
# Can't use RFE with an rbf kernel

# Create SVM with the best parameters
svmmodel = svm.SVC(C=100, gamma=0.1, kernel='rbf')
# Use with all features
fit_evaluate_cv(svmmodel, X_train, y_train)

# Finally apply final chosen model (SVM) to remaining 20% testing set
from sklearn.metrics import confusion_matrix, f1_score
svmtesting = svm.SVC(C=100, gamma=0.1, kernel='rbf', class_weight='auto')
svmtesting.fit(X_test, y_test)
y_pred = svmtesting.predict(X_test)

print("F1 score", f1_score(y_test, y_pred))
print("Confusion matrix", confusion_matrix(y_test, y_pred))
