from django.shortcuts import render
import pandas as pd
import numpy as np
from django.http import HttpResponse
import sklearn
import sklearn.model_selection 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


df6=pd.read_csv(r'D:\MCA\SEM-6\HousePrice_Prediction\df6.csv')


def initialize():
    dummies = pd.get_dummies(df6.Locality)
    dummies.drop('other',axis='columns',inplace=True)
    df7 = pd.concat([df6,dummies],axis='columns')
    df8 = df7.drop('Locality',axis='columns')
    dummies2 = pd.get_dummies(df8.Furnishing)
    df9 = pd.concat([df8,dummies2],axis='columns')
    df10 = df9.drop('Furnishing',axis='columns')
    df10.drop(columns='Type',inplace=True)
    x = df10.drop(['Price'],axis='columns')
    col=x.columns
    maincol=[]
    for i in range(5,len(col)-3):
        maincol.append(col[i])
    y=df10.Price
    return x,y,maincol

X,Y,maincol=initialize()

def predict_price(location,area,bhk,bath,park,Type):
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    lr_clf.fit(X_train,y_train)
    lr_clf.score(X_test,y_test)
    cv = ShuffleSplit(n_splits =5,test_size =0.2, random_state =0)
    loc_index = np.where(X.columns==location)[0][0]
    type_index = np.where(X.columns==Type)[0][0]    
    x= np.zeros(len(X.columns))
    x[0]= area
    x[1] = bhk
    x[2] = bath
    x[3] = park
    if loc_index >= 0:
        x[loc_index] =1
    if loc_index >= 0:
        x[type_index] =1        
    return lr_clf.predict([x])[0]


class SearchData:
  def __init__(self, area,bhk,bathroom,furn,Type, loc,park,price):
    self.area = area
    self.bhk=bhk
    self.bathroom=bathroom
    self.Furnishing=furn
    self.type=Type
    self.loc = loc
    self.park=park
    self.price=price


def index(request):
    return render(request,'index.html',{'msg':maincol})
    
def predict(request):
    return render(request,'predict.html',{'msg':maincol})

def searchLocation(request):
    return render(request,'searchLocation.html',{'msg':maincol})


def getdata(request):
    if request.method=="POST":
        Locn=request.POST.get('Locn')      
        tempdf=df6.loc[ df6['Locality'].str.contains(Locn)]
        AreaCol=list(tempdf["Area"])
        BHKCol=list(tempdf["BHK"])
        BathroomCol=list(tempdf["Bathroom"])
        FurnishingCol=list(tempdf["Furnishing"])
        TypeCol=list(tempdf["Type"])
        LocalityCol=list(tempdf["Locality"])
        ParkingCol=list(tempdf["Parking"])
        PriceCol=list(tempdf["Price"])
        searchResult=[]
        for i in range(len(AreaCol)):
            x = SearchData(AreaCol[i],BHKCol[i],BathroomCol[i],FurnishingCol[i],TypeCol[i],LocalityCol[i],ParkingCol[i],PriceCol[i])
            searchResult.append(x)
        return render(request,'getdata.html',{'data':searchResult})


def findPrice(request):
    if request.method=="POST":
        area=request.POST.get('area','')
        Location=request.POST.get('Location','')
        BHK=request.POST.get('BHK','')
        Bathroom=request.POST.get('Bathroom','')
        Parking=request.POST.get('Parking','')
        Type=request.POST.get('Type','')
        cost=predict_price(Location,area,BHK,Bathroom,Parking,Type)
        cost="Estimated Price is  Rs "+str(abs(cost))
    return render(request,'predict.html',{'price':cost})

