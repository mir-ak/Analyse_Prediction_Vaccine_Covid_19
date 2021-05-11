import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objs as go
import folium, geopy, os ,time
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
from prophet import Prophet
from prophet.plot import plot_plotly

#Suppression de certaines colonnes
def drop_columns(df):
    df.drop(['source_name','source_website'],inplace=True,axis=1)

#Formatage et fixation de la date
def format_set_date(df):
    df['date']= pd.to_datetime(df['date'])
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-04-28')
    df.loc[(df['date'] > start_date) & (df['date'] < end_date)]

#Remplissage des valeurs manquantes avec des 0 ou 'Unknow'
def missing_values(df):
    df = df.apply(lambda x:x.fillna(0) if x.dtypes.kind in 'biufc' else x.fillna('Unknow'))

#Affichage des vaccins utilisés dans le monde
def display_vaccines(df):
    #recent_date=df[df['date'] == '2021-04-28']
    vaccine_ordered = df['vaccines'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    sns.set_style("whitegrid")
    sns.set_context('paper')
    f, ax = plt.subplots(1, 1)
    sns.barplot(x = vaccine_ordered.index, y = vaccine_ordered , palette='rocket')
    ax.set_title("Vaccins utilisés dans le monde entier",fontsize=16)
    ax.set_ylabel('Nombre de vaccins',fontsize=10)
    plt.xticks(rotation=60)
    plt.tight_layout()
    #plt.show()
    plt.savefig('analyse/Vaccins_utilisés_dans_le_monde_entier.png')
    
#Retourne les informations de vaccination par jour de chaque pays
def country_daily_vaccinations(df):
    country_daily_vaccination =df[df['date']=='2021-04-28'].groupby(['country','iso_code','vaccines',])['daily_vaccinations'].sum().reset_index()
    return country_daily_vaccination

#Affichage des vaccins utilisés par différents pays dans une map
def display_vaccins_in_country_map(df):
    fig = px.choropleth(df, locations="iso_code",
        color="vaccines",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Plasma,
        title= "Vaccins utilisés par différents pays")
    fig.update_layout(showlegend=True)
    #fig.show()
    fig.write_html("analyse/Vaccins_utilises_par_différents_pays.html")

#Affichage des 30 meilleurs des pays les plus vaccinés
def display_top_50_daily_vaccination(df):
    country_daily_vaccination = country_daily_vaccinations(df)
    top_countries = country_daily_vaccination[['country','vaccines','daily_vaccinations']].sort_values('daily_vaccinations', ascending=False).reset_index(drop=True).head(50)  
    sns.set_style('whitegrid')
    sns.set_context('paper')
    f,ax = plt.subplots(1,1)
    ax= sns.barplot(y='country',x='daily_vaccinations',data=top_countries,palette='GnBu_d')
    ax.set_xlabel('Total vaccinations')
    ax.set_ylabel('Countries')
    ax.set_title('Top 50 des pays les plus vaccinés',fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos :'{:,.2f}'.format(x/100000)+'M'))
    plt.tight_layout()
    #plt.show()
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.savefig('analyse/Top_50_des_pays_les_plus_vaccines.png',dpi=250)

#Affichage du nombre de vaccinations par jour en France en 2021
def daily_vaccination_in_france(df):
    plt.figure(figsize=(14, 14))
    plt.subplot(1, 1, 1)
    sns.scatterplot(x='date',y='daily_vaccinations',data=df[df.country == 'France'])
    plt.title('Nombre de vaccinations par jour en France (2021)',fontweight="bold", size=20)
    #plt.show()
    plt.savefig('analyse/Nombre_de_vaccinations_par_jour_en_France_2021.png')
    

#Prediction : nb de personnes vaccinées par cent dans les 60 prochains jours (jusqu'au 28/06/21)
def predict_in_france(data, country, vaccination_metric, future_days, plot=True):
    df = data[(data['country'] == country)]
    df = df[['date', vaccination_metric]]
    df.columns = ['ds', 'y']
    model = Prophet(interval_width = 0.95)
    model.fit(df)
    layout = dict(title='Prediction : Nombre de personnes vaccinées par cent dans les 60 prochains jours (jusqu'"'"'au 28/06/21)',xaxis=dict(title = 'Dates'), yaxis=dict(title = 'Pourcentage'))
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    if plot:
        fig = plot_plotly(model, forecast)
        fig.layout = layout
        fig.write_html("analyse/Prediction_France.html")
    else:
        return forecast

#Permet la localisation d'un pays sur une carte
def findGeoCord(country):
    try:
        geolocator = Nominatim(user_agent='your_app_name')
        return geolocator.geocode(country)
    except GeocoderTimedOut:
        return findGeoCord(country)

#Création d'une page html affichant la carte avec le nombre total de vaccination par jour
def createMapToHtml(df):
    longitude = []
    latitude = []
    t = 115
    country_daily_vaccination = country_daily_vaccinations(df)
    for i in (country_daily_vaccination["country"]):
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        if findGeoCord(i) != None:
            loc = findGeoCord(i)
            longitude.append(loc.longitude)
            latitude.append(loc.latitude)
            print('le temps reste pour créer la map', timer, end="\r")
            t -= 1 
           
        else:
            longitude.append(np.nan)
            latitude.append(np.nan)     
    country_daily_vaccination['Longitude'] = longitude
    country_daily_vaccination['Latitude'] = latitude 
    f = folium.Figure(width=1000,height=500)
    map = folium.Map(tiles="cartodbpositron",max_bounds=True,min_zoom=1.5).add_to(f)
    marker_cluster = MarkerCluster().add_to(map)
    for i in range(len(country_daily_vaccination)):
        lon = country_daily_vaccination.iloc[i]['Longitude']
        lat = country_daily_vaccination.iloc[i]['Latitude']
        radius=5
        popup_text="""Country : {}<br> nombre total de vaccinations par jour {}<br>"""
        popup_text = popup_text.format(country_daily_vaccination.iloc[i]['country'],country_daily_vaccination.iloc[i]['daily_vaccinations'])
        folium.CircleMarker(location=[lat,lon],radius=radius,popup=popup_text,fill=True).add_to(marker_cluster)
    map.save('analyse/Map.html')
    print('Fin !, la map.html est créée vous y trouverez dans le dossier analyse ')
# créée dossier analyse
def createFolder():
    if not os.path.exists('analyse'):
        os.makedirs('analyse')  
def main():
    covid_vaccination = pd.read_csv('donnees/country_vaccinations.csv',sep=',')
    #nettoyage des données
    drop_columns(covid_vaccination)
    format_set_date(covid_vaccination)
    missing_values(covid_vaccination)
    createFolder()
    
    #appel des differentes fonctions d'analyse
    display_vaccines(covid_vaccination)
    display_vaccins_in_country_map(covid_vaccination)
    display_top_50_daily_vaccination(covid_vaccination)
    daily_vaccination_in_france(covid_vaccination)
    predict_in_france(covid_vaccination, 'France', 'people_vaccinated_per_hundred', 60)
    print("le dossier analyse est créée, vous y trouverez les résultats ")
    print("INFO : des fois quand vous ouvrez le dossier analyse il met de temps pour charge les donnees, il faut sortir et reouvrir le dossier a nouveau.")
    createMapToHtml(covid_vaccination)

if __name__ == '__main__':
    main()
