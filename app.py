from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()

#For standardizing features. We'll use the StandardScaler module.
from sklearn.preprocessing import StandardScaler

#Hierarchical clustering with the Sci Py library. We'll use the dendrogram and linkage modules.
from scipy.cluster.hierarchy import dendrogram, linkage
#Sk learn is one of the most widely used libraries for machine learning. We'll use the k means and pca modules.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# We need to save the models, which we'll use in the next section. We'll use pickle for that.
import pickle

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'sess.123'
CORS(app)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])

def upload():
    if request.method == 'POST':
        file = request.files['file']
        global df
        df = pd.read_csv(file, index_col=0)
        df.dropna(inplace=True)  # Remove rows with any missing values
        df.drop_duplicates(inplace=True)
        
        
        label_encoders = {}
        encoded_columns = ['Gender', 'Payment_method', 'Device_Type', 'Product_Category', 'Product', 
                        'Customer_Login_type', 'Order_Priority']

        for col in encoded_columns:
            label_encoders[col] = LabelEncoder()
            df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

        # # Remove original categorical columns
        df.drop(encoded_columns, axis=1, inplace=True)  


        plt.figure(figsize=(10, 7))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.savefig('static/images/heatmap.png')

        # Standardize the data
        scaler = StandardScaler()
        df_std = scaler.fit_transform(df)
        df_std = pd.DataFrame(data = df_std, columns=df.columns)

        # Create a PCA instance: pca

        wcss = []
        for i in range(1, 12):
            kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans_pca.fit(df_std)
            wcss.append(kmeans_pca.inertia_)
        
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, 12), wcss, marker='o', linestyle='--', color='red')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('static/images/elbow.png')

        kmeans = KMeans(n_clusters=12, init='k-means++', random_state=42)
        kmeans.fit(df_std)
        df_segm_kmeans = df_std.copy()
        df_segm_kmeans['Customer_Id'] = kmeans.labels_

        # Perform grouping and analysis on df_segm_kmeans, not df_std
        df_segm_analysis = df_segm_kmeans.groupby(['Customer_Id']).mean()
        df_segm_analysis

        column_order = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 
                'Gender_encoded', 'Payment_method_encoded', 'Device_Type_encoded', 
                'Product_Category_encoded', 'Product_encoded', 
                'Customer_Login_type_encoded', 'Order_Priority_encoded']
        df_segm_kmeans['labels_names'] = kmeans.labels_
        df_segm_analysis['labels_names'] = range(0,12)
        df_segm_analysis
        label_names = {
            0: 'Sales',
            1: 'Quantity',
            2: 'Discount',
            3: 'Profit',
            4: 'Shipping_Cost',
            5: 'Gender',
            6: 'Payment Method',
            7: 'Device Type',
            8: 'Product Category',
            9: 'Product',
            10: 'Customer Login Type',
            11: 'Order Priority',
        }
        # Create a new DataFrame with x and y coordinates, cluster labels, and label names
        x_axis = 'Sales'
        y_axis = 'Profit'
        df_plot = df_segm_kmeans[[x_axis, y_axis, 'labels_names']].copy()
        df_plot['label_names'] = df_plot['labels_names'].map(label_names)

        # Plot the scatterplot

        plt.figure(figsize=(10, 8))
        scatterplot = sns.scatterplot(data=df_plot, x=x_axis, y=y_axis, hue='label_names', hue_order=label_names.values(),palette=['green', 'orange', 'brown', 'dodgerblue', 'red', 'gray', 'pink', 'purple', 'cyan', 'yellow', 'black', 'blue'])
        plt.title('Segmentation K-means')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        # Set legend title to 'Cluster'
        legend = scatterplot.legend()
        legend.set_title("Cluster")

        plt.savefig('static/images/segmentation.png')

        pca = PCA()
        pca.fit(df_std)
        pca.explained_variance_ratio_

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')

        plt.savefig('static/images/pca.png')

        pca = PCA(n_components=12)
        pca.fit(df_std)

        pca.components_
        df_pca_comp = pd.DataFrame(data=pca.components_, columns=df_std.columns.values)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_pca_comp, annot=True, cmap='coolwarm')
        plt.savefig('static/images/pca_heatmap.png')

        pca.transform(df_std)
        scores_pca = pca.transform(df_std)

        wcss = []
        for i in range(1, 12):
            kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans_pca.fit(scores_pca)
            wcss.append(kmeans_pca.inertia_)

        plt.figure(figsize=(10, 7))
        plt.plot(range(1, 12), wcss, marker='o', linestyle='--', color='red')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('static/images/pca_elbow.png')

        kmeans_pca = KMeans(n_clusters=12, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)

        df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
        df_segm_pca_kmeans.columns.values[-12:] = ['Component 1', 'Component 2', 'Component 3','Component 4', 'Component 5', 'Component 6','Component 7', 'Component 8', 'Component 9','Component 10', 'Component 11', 'Component 12']
        df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

        df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
        df_segm_pca_kmeans_freq
        df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA', 'Sales']].groupby(['Segment K-means PCA']).count()
        df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
        df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0: 'Sales',
                                                                    1: 'Quantity',
                                                                    2: 'Discount',
                                                                    3: 'Profit',
                                                                    4: 'Shipping_Cost',
                                                                    5: 'Gender',
                                                                    6: 'Payment Method',
                                                                    7: 'Device Type',
                                                                    8: 'Product Category',
                                                                    9: 'Product',
                                                                    10: 'Customer Login Type',
                                                                    11: 'Order Priority'})
        df_segm_pca_kmeans_freq

        df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0: 'Sales',
                                                                             1: 'Quantity',
                                                                             2: 'Discount',
                                                                             3: 'Profit',
                                                                             4: 'Shipping_Cost',
                                                                             5: 'Gender',
                                                                             6: 'Payment Method',
                                                                             7: 'Device Type',
                                                                             8: 'Product Category',
                                                                             9: 'Product',
                                                                             10: 'Customer Login Type',
                                                                             11: 'Order Priority'})
        x_axis = 'Component 3'  # Change this to the desired PCA component column name
        y_axis = 'Component 1'  # Change this to the desired PCA component column name

        plt.figure(figsize=(10, 8))
        scatterplot = sns.scatterplot(data=df_segm_pca_kmeans, x=x_axis, y=y_axis, hue='Legend', palette=['green', 'orange', 'brown', 'dodgerblue', 'red', 'gray', 'pink', 'purple', 'cyan', 'yellow', 'black', 'blue'])
        plt.title('Clusters by PCA Components')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        # Set legend title to 'Cluster'
        legend = scatterplot.legend()
        legend.set_title("Cluster")

        plt.savefig('static/images/pca_segmentation.png')

    #     stored_file = session.get('uploaded_file')
    # if stored_file:
    #     print("Stored file path:", stored_file)
    #     # Use the stored file for further processing
    #     df = pd.read_csv(stored_file, index_col=0)
    #     print(df.head())  # Print the first few rows of the DataFrame

    #     numeric_columns = ['Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Sales']
    #     unique_products = df['Product'].unique()

    #     # Create a separate graph for each numeric column
    #     for numeric_col in numeric_columns:
    #         plt.figure(figsize=(10, 6))
            
    #         # Group data by product and calculate mean of the numeric column
    #         grouped_data = df.groupby('Product')[numeric_col].mean().reset_index()
            
    #         sns.barplot(data=grouped_data, x='Product', y=numeric_col)
    #         plt.title(f'{numeric_col} for Different Products')
    #         plt.xlabel('Product')
    #         plt.ylabel(numeric_col)
    #         plt.xticks(rotation=90)
    #     plt.savefig('static/images/total_sales_by_gender.png')

        plot_path = {
            'heatmap': 'static/images/heatmap.png',
            'elbow': 'static/images/elbow.png',
            'segmentation': 'static/images/segmentation.png',
            'pca': 'static/images/pca.png',
            'pca_heatmap': 'static/images/pca_heatmap.png',
            'pca_elbow': 'static/images/pca_elbow.png',
            'pca_segmentation': 'static/images/pca_segmentation.png',

        }
        return jsonify({'plot_path': plot_path})
    
@app.route('/upload-file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file is not None:
            session['uploaded_file'] = uploaded_file.filename
            return jsonify({'message': 'File uploaded successfully', 'filename': uploaded_file.filename})
        else:
            return jsonify({'error': 'No file uploaded'})


@app.route('/get_bar_plot_coordinates', methods=['GET'])
def get_bar_plot_coordinates():
    if request.method == 'GET':
        uploaded_file = session.get('uploaded_file')
        if uploaded_file is not None:
            uploaded_df = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/pmbackendflask/E-commerce Dataset.csv', index_col=0)
            numeric_columns = ['Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Sales']
            bar_coords_list = []

            for numeric_col in numeric_columns:
                # Group data by product and calculate mean of the numeric column
                grouped_data = uploaded_df.groupby('Product')[numeric_col].mean().reset_index()
                bar_coords = []

                for i, (_, row) in enumerate(grouped_data.iterrows()):
                    bar_coords.append({'x': row['Product'], 'y': row[numeric_col]})
                
                bar_coords_list.append({numeric_col: bar_coords})
            
            # Calculate total customers, total sales, and total profit
            total_customers = int(uploaded_df['Gender'].count())
            total_sales = int(uploaded_df['Sales'].sum())
            total_profit = int(uploaded_df['Profit'].sum())

            return jsonify({
                'bar_coords': bar_coords_list,
                'total_customers': total_customers,
                'total_sales': total_sales,
                'total_profit': total_profit
            })
        else:
            return jsonify({'error': 'No file uploaded'})


    


        # return jsonify({'bar_coords': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]})
    
    
if __name__ == "__main__":
    app.run(debug=True)





                                                                                