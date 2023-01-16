from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
city_encoded=pickle.load(open('city.pkl','rb'))


@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        city=request.form.get('City')
        city=city_encoded.transform([city])
        city=city.item()
        print(city)
        #city=np.reshape(city,(-1,1))

        PM25=float(request.values['PM2.5'])
        PM25=np.reshape(PM25,(-1,1))

        PM10=float(request.values['PM10'])
        PM10=np.reshape(PM10,(-1,1))

        NO=float(request.values['NO'])
        NO=np.reshape(NO,(-1,1))

        NO2=float(request.values['NO2'])
        NO2=np.reshape(NO2,(-1,1))
        
        NH3=float(request.values['NH3'])
        NH3=np.reshape(NH3,(-1,1))
        
        CO=float(request.values['CO'])
        CO=np.reshape(CO,(-1,1))
        
        SO2=float(request.values['SO2'])
        SO2=np.reshape(SO2,(-1,1))        
        
        O3=float(request.values['O3'])
        O3=np.reshape(O3,(-1,1))
        
        Benzene=float(request.values['Benzene'])
        Benzene=np.reshape(Benzene,(-1,1))
        
        Toluene=float(request.values['Toluene'])
        Toluene=np.reshape(Toluene,(-1,1))
        
        Xylene=float(request.values['Xylene'])
        Xylene=np.reshape(Xylene,(-1,1))
        
        details=[city,PM25,PM10,NO,NO2,NH3,CO,SO2,O3,Benzene,Toluene,Xylene]
        data_out=np.array(details).reshape(1,-1)
        output=model.predict(data_out)
        output=output.item()
        indexvalue=output
        if output<=50:
            output="Health impact is minimum"
        elif (output>50) and (output<=100) :
            output="Minor breathing discomfort to  sensitive people"
        elif (output>100) and (output<=200):
            output="Breathing discomfort to the people with lungs,asthma and heart diseases"
        elif (output>200) and (output<=300):
            output="Breathing discomfort to most people on prolonged exposure"
        elif (output>300) and (output<=400):
            output="Respiratory illness on prolonged exposure"
        else:
            output="Affects healthy people and seriously impacts those with existing diseases"
        #output=output.decode()
        return render_template('result.html',prediction_text='The AQI index value is {} for the corresponding the pollutants concentrations and the health impact is :{}'.format(indexvalue,output))



if __name__ == '__main__':
    app.run(port=5000)
    #app.run(debug=True)



