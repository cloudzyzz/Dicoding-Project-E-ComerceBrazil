import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import folium

# ---- Fungsi Utilitas ----
import pandas as pd
import requests

def load_data():
    """
    Memuat dataset dari Google Drive
    """
    try:
        # URL dataset dari Google Drive
        sales_reviews_url = "https://drive.google.com/file/d/1WbGpGDfakAtseeDJQC-fFjyldhAWECPS/view?usp=sharing"
        orders_merged_url = "https://drive.google.com/file/d/1NypuA2Qed_kyFSY4dE6dR9X7gyWGomvh/view?usp=sharing"
        product_reviews_url = "https://drive.google.com/file/d/19TcJk91HR5z2UO90LU8eDlkvJr5AMfMf/view?usp=sharing"
        merged_data_url = "https://drive.google.com/file/d/1gYM7j1dt-Zf7Glc51OpFz6Cas5RCul6y/view?usp=sharing"

        # Unduh dataset
        sales_reviews = pd.read_csv(sales_reviews_url)
        orders_merged = pd.read_csv(orders_merged_url)
        product_reviews = pd.read_csv(product_reviews_url)
        merged_data = pd.read_csv(merged_data_url)

        return sales_reviews, orders_merged, product_reviews, merged_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None


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
def create_sales_review_plots(sales_reviews):
    """
    Membuat visualisasi hubungan penjualan dan ulasan
    """
    seller_performance = sales_reviews.groupby('seller_id').agg({
        'order_item_id': 'count',
        'review_score': 'mean'
    }).reset_index().rename(columns={'order_item_id': 'total_sales'})

    # Seaborn plot
    fig_sns, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=seller_performance, x='total_sales', y='review_score', alpha=0.7, ax=ax)
    ax.set_title('Total Sales vs Average Review Score')
    ax.set_xlabel('Total Sales')
    ax.set_ylabel('Average Review Score')
    st.pyplot(fig_sns)

    # Plotly plot
    fig_plotly = px.scatter(seller_performance, 
                           x='total_sales', 
                           y='review_score',
                           title='Total Sales vs Review Score per Seller',
                           labels={'total_sales': 'Total Sales', 
                                 'review_score': 'Average Review Score'},
                           hover_data=['seller_id'])
    st.plotly_chart(fig_plotly)

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

    # Plotly plot
    fig_plotly = px.bar(payment_analysis, 
                        x='payment_type', 
                        y='processing_time',
                        title='Average Processing Time by Payment Method',
                        labels={'payment_type': 'Payment Method', 
                               'processing_time': 'Avg Processing Time (hours)'},
                        color='processing_time')
    st.plotly_chart(fig_plotly)

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
    sales_reviews, orders_merged, product_reviews, merged_data = load_data()
    
    # Periksa apakah semua data berhasil dimuat
    if all(df is not None for df in [sales_reviews, orders_merged, product_reviews, merged_data]):
        # Sidebar navigation
        st.sidebar.title('Navigation')
        selected_page = st.sidebar.radio(
            'Pilih Halaman',
            [
                'Analysis Awal',  # Berisi 3 visualisasi awal
                'Analysis Lanjutan'  # Berisi 2 visualisasi lanjutan
            ]
        )
        
        # Halaman "Analysis Awal"
        if selected_page == 'Analysis Awal':
            st.header('Analysis Awal')
            st.subheader('Hubungan Penjualan dan Ulasan')
            create_sales_review_plots(sales_reviews)
            
            st.subheader('Metode Pembayaran dan Waktu Pemrosesan')
            create_payment_analysis_plots(orders_merged)
            
            st.subheader('Kategori Produk dan Kepuasan Pelanggan')
            create_product_category_plots(product_reviews)

        # Halaman "Analysis Lanjutan"
        elif selected_page == 'Analysis Lanjutan':
            st.header('Analysis Lanjutan')
            st.subheader('RFM Analysis')
            rfm_analysis(merged_data)
            
            st.subheader('Clustering Analysis')
            clustering_analysis(merged_data)
    
    # Pesan jika data gagal dimuat
    else:
        st.error("Gagal memuat data. Pastikan file CSV tersedia dan sesuai dengan kebutuhan aplikasi.")

# Jalankan aplikasi
if __name__ == '__main__':
    main()
