import pandas as pd
import streamlit as st
from random import choices



st.set_page_config(layout="wide")




menu = ["Home","Data","Visuals","Prediction Tool"]
choice = st.sidebar.selectbox("Menu",menu)
menu = ["Home","Data","Visuals","Prediction Tool"]
if choice == "Home":
    st.title("Home Page")
    html_string = '<a href="https://im.ge/i/FLOGMP"><img src="https://i.im.ge/2022/07/25/FLOGMP.jpg" alt="FLOGMP.jpg" border="0"></a>'
    st.markdown(html_string,unsafe_allow_html= True)

if choice == "Data":
    st.markdown("<h1 style='text-align: center; color: green;'>Look into the Raw data</h1>", unsafe_allow_html=True)

    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')
    if uploaded_data:
        df= pd.read_csv(uploaded_data)
        col = df.columns
        Kpi1, Kpi2, Kpi3, Kpi4 =st.columns(4)
        Kpi1.metric(label ="Number of Columns", value = len(col))
        Kpi2.metric(label="Number of rows", value = len(df))
        Kpi3.metric(label="Null values", value = df.isnull().sum().sum())
        Kpi4.metric(label="Duplicates", value =df.duplicated().sum().sum())
        st.dataframe(df,width=1200, height=300)
	
if choice == "Visuals":
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')

    if uploaded_data:

        from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
        import plotly.express as px
        import plotly.graph_objects as go






        df=pd.read_csv(uploaded_data)
        st.markdown("<h1 style='text-align: center; color: green;'>Data Visualization</h1>", unsafe_allow_html=True)
        df8=df.copy()
        df8.fillna(0, inplace= True)
        desc = {"_CPE_":"CPE","_CPLPV_": "CPLPV","_CPC_":"CPC","_CPMR_":"CPMR","_CPV_":"CPV","_CPM_":"CPM","_CPL_":"CPL","_CPA_":"CPA","_Remarketing_":"Remarkerting"}

        def check_desc(x):
            for key in desc:
                if key.lower() in x.lower():
                    return desc[key]
            return ''

        df8["Campaign_objective"] = df8["Campaign name"].map(lambda x: check_desc(x))

        desc = {"_EGYPT_":"EGYPT","KSA":"KSA","_UAE_":"UAE","QATAR":"QATAR"}

        def check_desc(x):
            for key in desc:
                if key.lower() in x.lower():
                    return desc[key]
            return ''

        df8["country"] = df8["Campaign name"].map(lambda x: check_desc(x))

        df8.rename(columns={'Campaign name':'Campaign_name','Link clicks':'Link_clicks','Landing page views':'Landing_page_views','Post engagement':'Post_engagement','3-second video plays':'Three_second_video_plays','Post comments':'Post_comments','Post saves':'Post_saves','Post reactions':'Post_reactions','Post shares':'Post_shares','Video plays at 50%':'Video_plays_at50_percent','Amount Spent':'Amount_Spent'}, inplace = True)


        df8.drop(columns=['Reporting starts','Reporting ends','Campaign_name'], axis=1, inplace = True)


        col1, col2 = st.columns(2)
        with col1:
            fig2= px.scatter(df8, x='Impressions',y='Amount_Spent', size='Amount_Spent', color='Amount_Spent')
            fig2 = go.Figure(fig2)
            col1.plotly_chart(fig2)

        with col2:
            fig3 = px.scatter(df8, x='Reach', y='Amount_Spent',size='Amount_Spent', color='Amount_Spent')
            fig3 = go.Figure(fig3)
            col2.plotly_chart(fig3)

        colA, colB = st.columns(2)
        with colA:
            df_obj = df8.groupby('Campaign_objective')['Campaign_objective'].count().reset_index(name='counts')
            fig1 = px.bar(df_obj, x='Campaign_objective', y = 'counts', color='Campaign_objective', title = "total campaigns by objective", height=800, width=700)
            fig1=go.Figure(fig1)
            st.plotly_chart(fig1)

        with colB:
            df_c = df8.groupby('country')['country'].count().reset_index(name='counts')
            fig4 = px.bar(df_c, x='country', y = 'counts', color='country', title = "total campaigns by country", height = 800, width=700)
            fig4=go.Figure(fig4)
            colB.plotly_chart(fig4)
        


if choice == "Model Selection":
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.express as px
    import plotly.graph_objects as go
    st.markdown("<h1 style='text-align: center; color: green;'>Data Cleaning,Preprocessing, Model Selection and tuning</h1>", unsafe_allow_html=True)
    uploaded_data = st.file_uploader('Upload dataset', type='csv')
    st.write('Importing Libraries')
    st.code("""
import pandas as pd
import numpy as np

""")

    st.write("importing Data")
    st.code("#imported through upload on streamlit")
    
    st.write("Exploring Data Shape")
    st.code("""
df.shape""")

    st.write("exploring Data Columns and Null values")
    st.code("""
df.info()""")

    st.write("Filling NAN values with 0, in which imputing with mean or median is not logical and dropping them will remove a lot of data, loss of information")
    st.code("""
df = df.fillna(0)""")


if choice == "Prediction Tool":
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')
    df7 = pd.read_csv(uploaded_data)
    if uploaded_data:
        
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        df7.fillna(0, inplace = True)

        desc = {"_CPE_":"CPE","_CPLPV_": "CPLPV","_CPC_":"CPC","_CPMR_":"CPMR","_CPV_":"CPV","_CPM_":"CPM","_CPL_":"CPL","_CPA_":"CPA","_Remarketing_":"Remarkerting"}

        def check_desc(x):
            for key in desc:
                if key.lower() in x.lower():
                    return desc[key]
            return ''

        df7["Campaign_objective"] = df7["Campaign name"].map(lambda x: check_desc(x))

        desc = {"_EGYPT_":"EGYPT","KSA":"KSA","_UAE_":"UAE","QATAR":"QATAR"}

        def check_desc(x):
            for key in desc:
                if key.lower() in x.lower():
                    return desc[key]
            return ''

        df7["country"] = df7["Campaign name"].map(lambda x: check_desc(x))

        df7.rename(columns={'Campaign name':'Campaign_name','Link clicks':'Link_clicks','Landing page views':'Landing_page_views','Post engagement':'Post_engagement','3-second video plays':'Three_second_video_plays','Post comments':'Post_comments','Post saves':'Post_saves','Post reactions':'Post_reactions','Post shares':'Post_shares','Video plays at 50%':'Video_plays_at50_percent','Amount Spent':'Amount_Spent'}, inplace = True)


        df7.drop(columns=['Reporting starts','Reporting ends','Campaign_name'], axis=1, inplace = True)

        X = df7[['Reach', 'Impressions', 'Link_clicks', 'Landing_page_views',
       'Post_engagement', 'Three_second_video_plays', 'ThruPlays',
       'Post_comments', 'Post_saves', 'Post_reactions', 'Post_shares',
       'Video_plays_at50_percent', 'Leads',
       'Campaign_objective', 'country']]
        y = df7['Amount_Spent']

        num_vars = X.select_dtypes(include=['float', 'int']).columns.tolist()
        cat_vars = X.select_dtypes(include=['object']).columns.tolist()


        cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop = 'first')

        #Pipeline for categorical features
        cat_pipeline = Pipeline([
        ('encoding', cat_encoder),
        ])

        #Pipeline for numerical features
        num_pipeline = Pipeline([
        ('scaler', StandardScaler())])
        #Pipeline to apply on all columns
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_vars),
        ("cat", cat_pipeline, cat_vars),
        ])


        from sklearn.model_selection import train_test_split
        train_val_set, test_set = train_test_split(df7, test_size=0.2, random_state=111, shuffle = True)

        train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=111, shuffle = True)

        y_train = train_set['Amount_Spent']
        X_train = train_set.drop(['Amount_Spent'], axis=1)

        y_valid = val_set['Amount_Spent']
        X_valid = val_set.drop(['Amount_Spent'], axis=1)

        y_test = test_set['Amount_Spent']
        X_test = test_set.drop(['Amount_Spent'], axis=1)


        from sklearn.ensemble import RandomForestRegressor

        forest_reg = RandomForestRegressor(bootstrap=False, max_features=2, n_estimators=10)

        pipeline = Pipeline(steps=[('i', full_pipeline), ('m', forest_reg)])

        pipeline.fit(X_train,y_train)

        def user_report():
            

            
            st.header("Please choose Campaign objective")

            Campaign_objective= st.selectbox("Campaign Objective",('CPE', 'CPC', 'CPLPV', 'Remarkerting', 'CPMR', 'CPA', '', 'CPV',
       'CPL'))

            st.header("please choose campaign country")
            
            country = st.selectbox("country",('EGYPT', 'KSA', '', 'UAE', 'QATAR'))
            
            st.header("Enter the daily KPIs")

            Reach=st.number_input("Reach")

            Impressions = st.number_input("Impressions")

            Link_clicks=st.number_input("Link_clicks")

            Landing_page_views=st.number_input("Landing_page_views")

            Post_engagement=st.number_input("Post_engagement")

            Three_second_video_plays = st.number_input("Three_second_video_plays")

            ThruPlays = st.number_input("ThruPlays")

            Post_comments=st.number_input("Post_comments")

            Post_saves = st.number_input("Post_saves")


            Post_reactions = st.number_input("Post_reactions")


            Post_shares=st.number_input("Post_shares")

            Video_plays_at50_percent= st.number_input("Video_plays_at50_percent")

            Leads=st.number_input("Leads")

           


            user_report_data = {
            'Reach': Reach, 'Impressions': Impressions, 'Link_clicks':Link_clicks, 'Landing_page_views':Landing_page_views,
       'Post_engagement':Post_engagement, 'Three_second_video_plays':Three_second_video_plays, 'ThruPlays':ThruPlays,
       'Post_comments':Post_comments, 'Post_saves':Post_saves, 'Post_reactions':Post_reactions, 'Post_shares':Post_shares,
       'Video_plays_at50_percent':Video_plays_at50_percent, 'Leads':Leads, 'Campaign_objective':Campaign_objective,'country':country}



            report_data = pd.DataFrame(user_report_data, index=[0])
            
            return report_data

        user_data= user_report()
        
        
        prediction_M = pipeline.predict(user_data)

        st.markdown("<h1 style='text-align: center; color: green;'>The Predicted Spend according to entries is</h1>", unsafe_allow_html=True)

        
        st.header(prediction_M)
        st.title("+- 7.3")





       



        



            
            

    

