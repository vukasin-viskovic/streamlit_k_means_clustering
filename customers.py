import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_lottie import st_lottie
import json
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans


st.set_page_config(layout="wide")

st.header('K-means clustering app') ########################################### Header #################

"""
We can use this app to cluster our data into the desired number of clusters using
the K-Means clustering method.

You can learn more about this method [here](https://en.wikipedia.org/wiki/K-means_clustering).

You can upload and use your own dataset, or just go with the one I provided 
(more info on it can be found [here](https://www.kaggle.com/datasets/srolka/ecommerce-customers?datasetId=85008&searchQuery=k).

If you upload your own dataset, rest assured that it won't be stored anywhere. 
You can always inspect my source code [here](https://github.com/vuxvix/streamlit_k_means_clustering).
"""

uploaded_file = st.file_uploader("Upload CSV", type = ".csv") ################### Upload ###############

use_example_file = st.checkbox("Use example file", 
                               False, 
                               help = "Use in-built example file to demo the app")

## If CSV is NOT uploaded, but checkbox IS filled, use values from the example file
## and pass them down to the next IF block
if use_example_file:
    uploaded_file = "ecommerce_customers.csv"
    ab_default = ["variant"]
    result_default = ["converted"]

## Once "uploaded_file == True", we can move on with our app
if uploaded_file:
    
    df = pd.read_csv(uploaded_file)

    st.markdown("### Data preview")
    """
    Okay, let's start by taking a quick look at our dataset.
    
    We can use `pd.DataFrame.head()` method to quickly check its first 5 rows and get a first 
    impression of the data.
    """
    
    st.dataframe(df.head()) ## Printing top five rows of the df

    """
    The above output should already have provided us some understanding of the nature of our data.
    
    We can learn a bit more about it using `pd.DataFrame.describe()` method.
    
    It provides a nice overview into our dataset's numeric columns.
    """
    
    st.write(df.describe()) ## Printing out the output of the "df.describe()" method

    ## Some condidtional formatting 
    st.write(f"#### Our dataset has {df.shape[0]} rows and {df.shape[1]} columns.") 
    
    """
    Finally, let's explore the missing and unique values across our dataset.
    """

    def check_nunique_missing(df): ################################################################

        check_list = []

        ## For each column in our df
        for col in df.columns:

            dtypes = df[col].dtypes ## Save it's data type
            nunique = df[col].nunique() ## Save the count of unique values
            not_na = df[col].notna().sum() ## Count of not-na values
            sum_na = df[col].isna().sum() ## Count of NAs

            ## I'm creating a list of lists - with one "sublist" per each column of the original df
            check_list.append([col, dtypes, nunique, not_na, sum_na]) 

        df_check = pd.DataFrame(check_list) ## list (of lists) -> pd.DataFrame

        ## Setting appropriate column names
        df_check.columns = ['column', 'dtypes', 'nunique', 'not_na', 'sum_na'] 

        return df_check 
    
    ###############################################################################################

    st.write(check_nunique_missing(df))
    
    st.write('---') ###############################################################################
        
    """
    ### Time to create our K-Means clustering model
    """
         
    numerics = ['int16', 'int32', 'int64', 
                'float16', 'float32', 'float64']
    
    num_cols = list(df.select_dtypes(include=numerics).columns)
    
    columns_for_model = st.multiselect("Select 2 or 3 dimensions for clustering", num_cols)
            
    if len(columns_for_model) == 1:
        st.write("Please choose at one more variable")
    
    elif (len(columns_for_model) == 2) or (len(columns_for_model) == 3):
    
        var1 = columns_for_model[0]
        var2 = columns_for_model[1]
        X = df.loc[:, [var1, var2]].values
        
        if len(columns_for_model) == 3:
            var3 = columns_for_model[2]
            X = df.loc[:, [var1, var2, var3]].values
        
                   
        """ 
        Now, we will use sklearn for our clustering.
        
        There's several parameters we can choose here.
        
        The first one is the **initialization method**.
        """
            
        chosen_initialization_method = st.selectbox('Choose initialization method', 
                                                    ['k-means++', 'random'])
                    
        """
        The second is the **number of clusters**.
        
        We can try out grouping our data into different numbers of clusters, and see
        where we'll get the best results.
        
        Let's specify the maximum number of clusters we want to try out.
        
        And let's observe how the algorithm performs with number of clusters ranging from 1 
        to our chosen maximum number of clusters (including it).
        """
         
        max_clusters = st.slider("Choose up to how many clusters to try out", 10, 20, 15)
        
        with st.echo():
            
            wcss = []
            
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters = i, 
                                init = chosen_initialization_method, 
                                random_state = 101)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
               
            fig, ax = plt.subplots()
            ax = plt.plot(range(1, max_clusters + 1), wcss)
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS values')
                
        with st.spinner('Training the model...'):
            time.sleep(5)
        st.success('Done!')
                
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig)
                    
        """
        Based on the elbow plot, let's now settle on the final **number of clusters**.
        
        We should choose the number at which the curve at the above plot starts to flatten.
        """

        num_clusters = st.slider("Decide on the final number of clusters", 1, 20, 5) 
        
        st.write("Now, let's do some plotting")
                                 
        kmeansmodel = KMeans(n_clusters = num_clusters, 
                            init = chosen_initialization_method, 
                            random_state = 101)
        y_kmeans = kmeansmodel.fit_predict(X)
                
        if len(columns_for_model) == 2: #########################################################
            
            col_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']    
            fig, ax = plt.subplots()
            
            for i in range(num_clusters):
                ax = plt.scatter(x = X[y_kmeans == i, 0], 
                                y = X[y_kmeans == i, 1], 
                                s = 80, 
                                c = col_list[i], 
                                label = f"Cluster {i + 1}")
            
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5)
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f"Our {num_clusters} clusters")
            plt.show()
        
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig)
        
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Done!')
                
        elif len(columns_for_model) == 3: ######################################################
                                    
            df_3d = pd.DataFrame({"x" : [],
                                    "y" : [],
                                    "z" : [],
                                    "cluster" : i})
            
            for i in range(num_clusters):
                mini_df = pd.DataFrame({"x" : X[y_kmeans == i, 0],
                                        "y" : X[y_kmeans == i, 1],
                                        "z" : X[y_kmeans == i, 2],
                                        "cluster" : i + 1})
                df_3d = df_3d.append(mini_df)
            
            df_3d = df_3d.astype({"cluster":'category'})    

            fig = px.scatter_3d(df_3d, 
                                x = "x", 
                                y = "y", 
                                z = "z", 
                                color = "cluster",
                                labels = {"x": var1,
                                        "y": var2,
                                        "z": var3},
                                title = "Clusters :)",
                                height = 800)
            
            st.plotly_chart(fig, use_container_width = True)
        
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Done!')

    elif len(columns_for_model) >= 3:
        """    
        To enable graphing, this app doesn't support clustering across more than 3 dimensions.
        
        Please remove the excess variables, to be able to successfully perform K-Means clustering :)      
        """
                
with open('99430-statistics.json', "r") as f: ########################################################
    data = json.load(f)
st_lottie(data, height = 300)
