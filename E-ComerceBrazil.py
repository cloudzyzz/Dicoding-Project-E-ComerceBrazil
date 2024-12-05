import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import folium

# ---- Fungsi Utilitas ----
def load_data():
    """
    Memuat dataset yang diperlukan
    """
    try:
        # Update to direct download links
        sales_reviews_url = "https://drive.google.com/uc?id=1WbGpGDfakAtseeDJQC-fFjyldhAWECPS"
        orders_merged_url = "https://drive.google.com/uc?id=1NypuA2Qed_kyFSY4dE6dR9X7gyWGomvh"
        product_reviews_url = "https://drive.google.com/uc?id=19TcJk91HR5z2UO90LU8eDlkvJr5AMfMf"
        merged_data_url = "https://drive.google.com/uc?id=1gYM7j1dt-Zf7Glc51OpFz6Cas5RCul6y"
        geo_orders_url = "https://drive.google.com/uc?id=1tcJ2MR2tEklI2LmMLl2YcolKJZaZSWuI"

        # Unduh dataset
        sales_reviews = pd.read_csv(sales_reviews_url)
        orders_merged = pd.read_csv(orders_merged_url)
        product_reviews = pd.read_csv(product_reviews_url)
        merged_data = pd.read_csv(merged_data_url)
        geo_orders = pd.read_csv(geo_orders_url)

        return sales_reviews, orders_merged, product_reviews, merged_data, geo_orders
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def filter_data_by_date(data, date_column, start_date, end_date):
    # Ensure date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    # Ensure start_date and end_date are datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Apply the filter
    return data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]

def process_orders_data(orders_merged):
    """
    Memproses data orders untuk analisis
    """
    orders_merged['order_approved_at'] = pd.to_datetime(orders_merged['order_approved_at'])
    orders_merged['order_purchase_timestamp'] = pd.to_datetime(orders_merged['order_purchase_timestamp'])
    orders_merged['processing_time'] = abs(
        (orders_merged['order_approved_at'] - orders_merged['order_purchase_timestamp']).dt.total_seconds() / 3600
    )
    return orders_merged

# ---- Fungsi Visualisasi ----
def create_sales_review_plots(sales_reviews, orders_merged):
    """
    Function that uses both sales_reviews and orders_merged
    """
    # Example of using orders_merged alongside sales_reviews
    seller_performance = sales_reviews.groupby('seller_id').agg({
        'order_item_id': 'count',
        'review_score': 'mean'
    }).reset_index().rename(columns={'order_item_id': 'total_sales'})

    # Generate plots (unchanged)
    fig_sns, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=seller_performance, x='total_sales', y='review_score', alpha=0.7, ax=ax)
    ax.set_title('Total Sales vs Average Review Score')
    ax.set_xlabel('Total Sales')
    ax.set_ylabel('Average Review Score')
    st.pyplot(fig_sns)

def create_payment_analysis_plots(orders_merged):
    """
    Membuat visualisasi analisis pembayaran
    """
    payment_analysis = orders_merged.groupby('payment_type').agg({
        'processing_time': 'mean',
        'order_id': 'count'
    }).reset_index().rename(columns={'order_id': 'total_orders'})

    # Seaborn plot
    fig_sns, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=payment_analysis, x='payment_type', y='processing_time', palette='viridis', ax=ax)
    ax.set_title('Processing Time by Payment Method')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Avg Processing Time (hours)')
    plt.xticks(rotation=45)
    st.pyplot(fig_sns)

def create_product_category_plots(product_reviews):
    """
    Membuat visualisasi kategori produk dan kepuasan pelanggan
    """
    category_analysis = product_reviews.groupby('product_category_name').agg({
        'review_score': ['mean', 'count']
    }).reset_index()
    category_analysis.columns = ['product_category_name', 'avg_review_score', 'total_reviews']

    # Top 10 categories
    top_categories = category_analysis.sort_values('avg_review_score', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_categories, x='avg_review_score', y='product_category_name', palette='viridis', ax=ax)
    ax.set_title('Top 10 Product Categories by Average Review Score')
    ax.set_xlabel('Average Review Score')
    ax.set_ylabel('Product Category')
    st.pyplot(fig)

def analyze_delivery_times(geo_orders):
    """
    Analyze delivery times by city and create visualizations
    """
    # Menghitung delivery time
    geo_orders['delivery_time'] = (
        pd.to_datetime(geo_orders['order_delivered_customer_date']) - pd.to_datetime(geo_orders['order_purchase_timestamp'])
    ).dt.total_seconds() / 3600

    # Aggregate delivery time by city and include geolocation data
    city_delivery = geo_orders.groupby('geolocation_city').agg({
        'delivery_time': 'mean',
        'order_id': 'count',
        'geolocation_lat': 'first',
        'geolocation_lng': 'first'
    }).reset_index().rename(columns={'order_id': 'total_orders'})

    # Identify fastest and slowest delivery cities
    fastest_city = city_delivery.loc[city_delivery['delivery_time'].idxmin()]
    slowest_city = city_delivery.loc[city_delivery['delivery_time'].idxmax()]

    # Bar plot: Average Delivery Time by City
    top_cities = city_delivery.sort_values('delivery_time').head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_cities, x='delivery_time', y='geolocation_city', palette='coolwarm')
    plt.title('Top 10 Cities with Fastest Delivery Times')
    plt.xlabel('Avg Delivery Time (hours)')
    plt.ylabel('City')
    st.pyplot(plt)
    plt.close()

    return city_delivery

# ---- Fungsi Analisis Lanjutan ----
def rfm_analysis(merged_data):
    """
    Melakukan analisis RFM
    """
    # Validasi kolom yang diperlukan
    required_columns = ['order_purchase_timestamp', 'customer_unique_id', 'order_id', 'price']
    missing_columns = [col for col in required_columns if col not in merged_data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan di data: {missing_columns}")
        return
    
    # Konversi kolom 'order_purchase_timestamp' ke datetime
    merged_data['order_purchase_timestamp'] = pd.to_datetime(
        merged_data['order_purchase_timestamp'], errors='coerce'
    )
    
    # Tambahkan kolom 'month'
    merged_data['month'] = merged_data['order_purchase_timestamp'].dt.to_period('M')
    
    # Hitung Recency, Frequency, dan Monetary
    rfm_monthly = merged_data.groupby(['month', 'customer_unique_id']).agg({
        'order_purchase_timestamp': lambda x: (pd.Timestamp.now() - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'price': 'sum'  # Monetary
    }).reset_index()
    
    # Rename kolom
    rfm_monthly.rename(columns={
        'order_purchase_timestamp': 'recency',
        'order_id': 'frequency',
        'price': 'monetary'
    }, inplace=True)
    
    # Konversi kolom ke numeric dengan eksplisit
    rfm_monthly['recency'] = pd.to_numeric(rfm_monthly['recency'], errors='coerce')
    rfm_monthly['frequency'] = pd.to_numeric(rfm_monthly['frequency'], errors='coerce')
    rfm_monthly['monetary'] = pd.to_numeric(rfm_monthly['monetary'], errors='coerce')
    
    # Isi nilai NaN dengan 0
    rfm_monthly.fillna(0, inplace=True)
    
    # Konversi kolom 'month' ke timestamp
    rfm_monthly['month'] = rfm_monthly['month'].dt.to_timestamp()
    
    # Rata-rata RFM per bulan
    rfm_monthly_avg = rfm_monthly.groupby('month')[['recency', 'frequency', 'monetary']].mean().reset_index()
    
    # Visualisasi menggunakan Matplotlib
    import matplotlib.pyplot as plt
    
    # Bersihkan plot sebelumnya
    plt.clf()
    
    # Plot Recency
    plt.figure(figsize=(10, 5))
    plt.plot(rfm_monthly_avg['month'], rfm_monthly_avg['recency'], marker='o')
    plt.title('Rata-rata Recency per Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Recency (Hari)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()
    
    # Plot Frequency
    plt.figure(figsize=(10, 5))
    plt.plot(rfm_monthly_avg['month'], rfm_monthly_avg['frequency'], marker='o')
    plt.title('Rata-rata Frequency per Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Frequency (Jumlah Order)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()
    
    # Plot Monetary
    plt.figure(figsize=(10, 5))
    plt.plot(rfm_monthly_avg['month'], rfm_monthly_avg['monetary'], marker='o')
    plt.title('Rata-rata Monetary per Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Monetary (Total Penjualan)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()
    
    return rfm_monthly_avg


def clustering_analysis(merged_data):
    """
    Melakukan analisis clustering berdasarkan perilaku customer
    """
    # Agregasi data customer
    customer_agg = merged_data.groupby('customer_unique_id').agg({
        'order_id': 'count',  # Jumlah transaksi
        'price': ['count', 'mean', 'sum']  # Analisis nilai transaksi
    }).reset_index()

    # Flatten kolom multiindex
    customer_agg.columns = [
        'customer_unique_id', 
        'transaction_count', 
        'unique_product_count', 
        'avg_order_value', 
        'total_spending'
    ]

    # Tambahkan kolom review score jika tersedia
    if 'review_score' in merged_data.columns:
        review_agg = merged_data.groupby('customer_unique_id')['review_score'].mean().reset_index()
        review_agg.columns = ['customer_unique_id', 'avg_review_score']
        customer_agg = pd.merge(customer_agg, review_agg, on='customer_unique_id', how='left')
    else:
        customer_agg['avg_review_score'] = 0  # Default jika tidak ada review score

    # Pilih fitur untuk clustering
    clustering_features = ['transaction_count', 'avg_order_value', 'total_spending']
    
    # Cek apakah kolom review score ada
    if 'avg_review_score' in customer_agg.columns:
        clustering_features.append('avg_review_score')
    
    # Preprocessing data
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    
    # Buat pipeline preprocessing
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Ganti NaN dengan median
        ('scaler', StandardScaler())  # Normalisasi
    ])
    
    # Persiapkan data untuk clustering
    features_to_cluster = customer_agg[clustering_features]
    
    # Preprocessing data
    features_preprocessed = preprocessor.fit_transform(features_to_cluster)
    
    # Clustering menggunakan KMeans
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_agg['cluster'] = kmeans.fit_predict(features_preprocessed)
    
    # Visualisasi cluster
    import matplotlib.pyplot as plt
    
    # Bersihkan plot sebelumnya
    plt.clf()
    
    # Plot distribusi cluster
    plt.figure(figsize=(10, 6))
    cluster_dist = customer_agg['cluster'].value_counts()
    cluster_dist.plot(kind='bar')
    plt.title('Distribusi Jumlah Customer per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Jumlah Customer')
    st.pyplot(plt)
    plt.close()
    
    # Analisis karakteristik cluster
    cluster_summary = customer_agg.groupby('cluster')[clustering_features].mean()
    st.write("Ringkasan Karakteristik Cluster:")
    st.write(cluster_summary)
    
    # Scatter plot untuk memvisualisasikan cluster
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        customer_agg['transaction_count'], 
        customer_agg['avg_order_value'], 
        c=customer_agg['cluster'], 
        cmap='viridis'
    )
    plt.title('Clustering Customer Berdasarkan Transaksi dan Nilai Order')
    plt.xlabel('Jumlah Transaksi')
    plt.ylabel('Rata-rata Nilai Order')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(plt)
    plt.close()
    
    return customer_agg

# ---- Main App ----
def main():
    st.title('Dashboard Analisis E-commerce')

    # Load data
    sales_reviews, orders_merged, product_reviews, merged_data , geo_orders = load_data()
    
    # Periksa apakah semua data berhasil dimuat
    if all(df is not None for df in [sales_reviews, orders_merged, product_reviews, merged_data, geo_orders]):
        # Sidebar navigation dan date range picker
        st.sidebar.title('Navigation')
        
        # Ensure datetime conversion
        orders_merged['order_purchase_timestamp'] = pd.to_datetime(orders_merged['order_purchase_timestamp'])
        
        # Date range picker
        min_date = orders_merged['order_purchase_timestamp'].min()
        max_date = orders_merged['order_purchase_timestamp'].max()
        
        start_date = st.sidebar.date_input(
            'Tanggal Mulai', 
            min_value=min_date.date(), 
            max_value=max_date.date(), 
            value=min_date.date()
        )
        
        end_date = st.sidebar.date_input(
            'Tanggal Akhir', 
            min_value=min_date.date(), 
            max_value=max_date.date(), 
            value=max_date.date()
        )
        
        # Convert date inputs to datetime for filtering
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date)
        
        # Filter data berdasarkan rentang tanggal
        filtered_orders_merged = filter_data_by_date(
            orders_merged, 
            'order_purchase_timestamp', 
            start_datetime, 
            end_datetime
        )
        
        # Filter merged_data
        filtered_merged_data = filter_data_by_date(
            merged_data, 
            'order_purchase_timestamp', 
            start_datetime, 
            end_datetime
        )
        
        # Sidebar navigation
        selected_page = st.sidebar.radio(
            'Pilih Halaman',
            [
                'Analysis Awal',  
                'Analysis Lanjutan'  
            ]
        )
        
        # Halaman "Analysis Awal"
        if selected_page == 'Analysis Awal':
            st.header('Analysis Awal')
            st.subheader('Hubungan Penjualan dan Ulasan')
            filtered_sales_reviews = sales_reviews[
                sales_reviews['order_id'].isin(filtered_orders_merged['order_id'])
            ]
            create_sales_review_plots(filtered_sales_reviews, filtered_orders_merged)
            
            st.subheader('Metode Pembayaran dan Waktu Pemrosesan')
            create_payment_analysis_plots(filtered_orders_merged)
            
            st.subheader('Kategori Produk dan Kepuasan Pelanggan')
            filtered_product_reviews = product_reviews[
            product_reviews['order_id'].isin(filtered_orders_merged['order_id'])
            ]
            create_product_category_plots(filtered_product_reviews)

            st.subheader('Kota yang Memiliki Waktu Pengiriman Tercepat')
            
            # Periksa apakah geo_orders tersedia
            if geo_orders is not None:
                # Filter geo_orders berdasarkan rentang tanggal
                filtered_geo_orders = filter_data_by_date(
                    geo_orders, 
                    'order_purchase_timestamp', 
                    start_datetime, 
                    end_datetime
                )
                
                # Jalankan analisis waktu pengiriman
                analyze_delivery_times(filtered_geo_orders)

        # Halaman "Analysis Lanjutan"
        elif selected_page == 'Analysis Lanjutan':
            st.header('Analysis Lanjutan')
            st.subheader('RFM Analysis')
            rfm_analysis(filtered_merged_data)
            
            st.subheader('Clustering Analysis')
            clustering_analysis(filtered_merged_data)
    
    # Pesan jika data gagal dimuat
    else:
        st.error("Gagal memuat data. Pastikan file CSV tersedia dan sesuai dengan kebutuhan aplikasi.")

# Jalankan aplikasi
if __name__ == '__main__':
    main()
