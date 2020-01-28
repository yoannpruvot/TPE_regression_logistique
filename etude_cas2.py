# Lucien Mauffret et Yoann Pruvot
#%%
# Veuillez changer le chemin suivant:
chemin="/home/yoann/Documents/Python/TPE_projet/Final/data_bank.csv"
#%%
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc, minimize
from scipy.stats import chi2
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def somme_poids(theta, x):
    return np.dot(x, theta)
def probabilite(theta, x):
    return sigmoid(somme_poids(theta, x))
def fonction_ll(theta, x, y):
    cout = -1 * np.sum(
        y * np.log(probabilite(theta, x)) + (1 - y) * np.log(
            1 - probabilite(theta, x)))
    return cout
def gradient(theta, x, y):
    return   np.dot(x.T, sigmoid(somme_poids(theta,   x)) - y)
def hessienne(theta,x,y):
    PIunmoinsPI=np.dot(np.diag(sigmoid(somme_poids(theta,x)).flatten()),np.diag(1-sigmoid(somme_poids(theta,x)).flatten()))
    return np.dot(np.dot(PIunmoinsPI,x).T,x)
def algorithme(x, y, theta):
    newton = fmin_tnc(func=fonction_ll, x0=theta,
                           fprime=gradient, args=(x, y.flatten()))
    return newton[0]
def algorithme2(x,y,theta):
    def t_gradient(param):
        return gradient(param,x,y)
    def t_hessienne(param):
        return hessienne(param,x,y)
    return Newton_system(t_gradient, t_hessienne,theta,10**-6)[0].flatten()
def prediction(x):
    theta = resultats[:, np.newaxis]
    return probabilite(theta, x)
def precision(x, reelle, seuil=0.5):
    predite = (prediction(x) >=
                         seuil).astype(int)
    predite = predite.flatten()
    pourcentage_precision = np.mean(predite == reelle)
    return pourcentage_precision * 100
def Newton_system(F, J, x, eps):
    """
    On veut résoudre l'équation F=0 avec la méthode de Newton.
    On note J la matrice Jacobienne de F. F et J sont des fonctions de x (un vecteur).
    En entrée, x prend la valeur du point auquel on veut commencer. 
    On s'arrête lorsque ||F||<eps ou le nb d'itérations est à 100.
    """
    F_value = F(x)
    F_norm = np.linalg.norm(F_value, ord=2)  # norme L2 d'un vecteur
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < 100:
        # On veut résoudre l'équation en h: F(x)+J(x)*h=0 (formule de la tangente)
        h = np.linalg.solve(J(x), -F_value)
        x = x + h
        F_value = F(x)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1
    # Soit on a trouvé une solution soit le compte est à -1 en sortie
    if abs(F_norm) > eps:
        iteration_counter = -1
    return(x, iteration_counter)

#%%
# Importation des données
# On va ici étudier le fait de faire du sport.
data=pd.read_csv(chemin, sep=",",index_col=False)
#%%
nb_ind_tot=len(data)
nb_oui=np.histogram(data['y'], bins=2)[0][1]
nb_non=np.histogram(data['y'], bins=2)[0][0]
print('La proportion totale de personne souscrivant au produit est: {:.2f}%'.format(100*nb_sport/nb_ind_tot))
#%%
data_vars=data.columns
#%%
cat_vars=['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data=data.join(cat_list)
#%%    
# On enlève les variables correspondant aux dummies créés
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data=data[to_keep]
#%%
#On sépare nos données en deux variables X (étudiée) et y (cible):
col_names=list(data.columns)
X_col_names=col_names
X_col_names.remove('y')
X=data[X_col_names].astype(int)
y=data['y'].astype(int)
#%%
# On met en place X = (1,X) et on rajoute un dimension à y, et on prépare notre theta
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))
# On découpe les données en 2 échantillons
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
#%%
# On lance notre algorithme
resultats = algorithme(X_train, y_train, theta)
print("Les paramètres trouvés sont :", resultats)
print("La précision de notre modèle est :", precision(X_test, y_test.flatten()))
#%%
# On crée notre matrice de confusion:
C_obs=np.dot(X_test,resultats)
for i in range(len(C_obs)):
    if C_obs[i]>0:
        C_obs[i]=1
    else:
        C_obs[i]=0
vrai_vrai=0
vrai_faux=0
faux_vrai=0
faux_faux=0
for i in range(len(C_obs)):
    if int(C_obs[i])==1:
        if y_test[i]==1:
            vrai_vrai=vrai_vrai+1
        else:
            faux_vrai=faux_vrai+1
    else:
        if y_test[i]==1:
            vrai_faux=vrai_faux+1
        else:
            faux_faux=faux_faux+1
mat_confusion=pd.DataFrame([[faux_faux,faux_vrai],[vrai_faux,vrai_vrai]])
mat_confusion.index=['faux_reel','vrai_relle']
mat_confusion.columns=['faux_calc','vrai_calc']
print(mat_confusion)
#%%
# Bootstrap afin de calculer l'écart type empirique:
Bootstrap = []
nb_sample = 100
resultat_general = []

for i in range(nb_sample):
    Bootstrap.append(resample(data))

for i in range(nb_sample):
    # On met en place nos X et y
    X_strap = Bootstrap[i][X_col_names]
    y_strap = Bootstrap[i]['y']
    # On met en place X = (1,X) et on rajoute un dimension à y, et on prépare notre theta
    X_strap = np.c_[np.ones((X_strap.shape[0], 1)), X_strap]
    y_strap = y_strap[:, np.newaxis]
    theta_strap = np.zeros((X_strap.shape[1], 1))
    resultat_general.append(algorithme(X_strap, y_strap, theta_strap))

std_theta = pd.DataFrame(resultat_general).std()

print('la variance est:',std_theta*std_theta)
#%%
# test de Wald
indice_bon = []
for i in range(1,len(theta)):
    T = resultats[i]*resultats[i]/(std_theta[i]*std_theta[i])
    if T >= chi2.ppf(0.95,1):
        indice_bon.append(X_col_names[i-1])
#%%
X=data[indice_bon].astype(int)
y=data['y'].astype(int)
#%%
# On met en place X = (1,X) et on rajoute un dimension à y, et on prépare notre theta
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))
# On découpe les données en 2 échantillons
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
#%%
# On lance notre algorithme
resultats = algorithme(X_train, y_train, theta)
print("Les paramètres trouvés sont :", resultats)
print("La précision de notre modèle est :", precision(X_test, y_test.flatten()))
