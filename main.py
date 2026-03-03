import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(comment):
    if isinstance(comment, str):
        return sia.polarity_scores(comment)['compound']
    else:
        return 0.0

# Function to generate recommendations
def generate_recommendations(model, fb_data):
    predicted_sales = model.predict(fb_data[['Likes', 'Shares', 'Comments', 'Reach', 'EngagementRate']])
    
    recommendations = []
    
    # Content Strategy
    if 'PostType' in fb_data.columns:
        high_engagement_posts = fb_data[fb_data['EngagementRate'] > fb_data['EngagementRate'].mean()]
        popular_post_types = high_engagement_posts['PostType'].mode().values
        recommendations.append(f"Focus on creating more {', '.join(popular_post_types)} posts.")
    else:
        recommendations.append("PostType column not found. Cannot provide specific content strategy recommendations.")
    
    recommendations.append("Boost posts with high engagement metrics to reach a wider audience. This can help increase visibility and attract new followers.")
    
    recommendations.append("Use relevant hashtags like #Oysters, #SustainableSeafood, and #LouisianaOysters in your posts to increase discoverability and reach users interested in these topics.")
    
    return recommendations

# Streamlit UI
st.title('Social Media Analytics for Oyster Marketing')
st.write('Upload your sales data and Facebook Insights data to get started.')

# File upload
sales_file = st.file_uploader('Upload Sales Data (Excel)', type='xlsx')
fb_file = st.file_uploader('Upload Facebook Insights Data (Excel)', type='xlsx')

if sales_file and fb_file:
    # Load data
    sales_data = pd.read_excel(sales_file)
    fb_data = pd.read_excel(fb_file)
    
    # Data cleaning
    sales_data.dropna(inplace=True)
    fb_data.dropna(inplace=True)
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    fb_data['Date'] = pd.to_datetime(fb_data['Date'])
    
    # Ensure Comments are strings
    if 'Comments' in fb_data.columns:
        fb_data['Comments'] = fb_data['Comments'].astype(str)
        fb_data['Sentiment'] = fb_data['Comments'].apply(analyze_sentiment)
    
    # Merge data
    merged_data = pd.merge(sales_data, fb_data, on='Date')
    
    # Ensure only numeric columns are used for correlation and regression
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    
    analyze_button = st.button('Analyze Data')
    back_button = st.button('Enter New Data')
    
    if analyze_button:
        if not merged_data[numeric_columns].empty:
            # Correlation and regression analysis
            correlation_matrix = merged_data[numeric_columns].corr()
            
            X = merged_data[['Likes', 'Shares', 'Comments', 'Reach', 'EngagementRate']].dropna()
            y = merged_data.loc[X.index, 'SalesVolume']
            
            if not X.empty and len(y) > 0:
                model = LinearRegression().fit(X, y)
                
                # Visualization
                st.subheader('Sales Volume Over Time')
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Calculate rolling average for smoothing
                merged_data['SmoothedSales'] = merged_data['SalesVolume'].rolling(window=2).mean()

                # Plot the original sales volume with markers
                ax.plot(merged_data['Date'], merged_data['SalesVolume'], marker='o', color='green', label='Actual Sales Volume', linewidth=2, linestyle='--')
                
                # Plot the smoothed sales volume
                ax.plot(merged_data['Date'], merged_data['SmoothedSales'], color='blue', label='Smoothed Sales Volume', linewidth=3)
                
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Sales Volume', fontsize=14)
                ax.set_title('Sales Volume Over Time', fontsize=18)
                ax.tick_params(axis='x', labelrotation=45, labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.grid(True)
                
                # Annotate maximum and minimum points
                max_sales = merged_data['SalesVolume'].max()
                max_date = merged_data.loc[merged_data['SalesVolume'].idxmax(), 'Date']
                min_sales = merged_data['SalesVolume'].min()
                min_date = merged_data.loc[merged_data['SalesVolume'].idxmin(), 'Date']
                
                
                # Add a legend
                ax.legend(loc='upper left')
                st.pyplot(fig)
                
                st.subheader('Correlation Matrix')
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                ax.set_title('Correlation Matrix', fontsize=18)
                st.pyplot(fig)
                
                # Bar chart for correlation values with 'SalesVolume'
                st.subheader("Correlations with Sales Volume")
                sales_corr = correlation_matrix['SalesVolume'].drop('SalesVolume')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sales_corr.plot(kind='bar', color='skyblue', ax=ax)
                ax.set_title('Correlation of Various Factors with Sales Volume', fontsize=18)
                ax.set_xlabel('Factors', fontsize=14)
                ax.set_ylabel('Correlation Coefficient', fontsize=14)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.tick_params(axis='x', labelrotation=45, labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                st.pyplot(fig)
                
                # Generate recommendations
                recommendations = generate_recommendations(model, fb_data)
                st.subheader('Recommendations')
                for rec in recommendations:
                    st.write(rec)
            else:
                st.write('Insufficient data for regression analysis. Please ensure there are enough data points with complete information.')
        else:
            st.write('No numeric data available for analysis. Please check your uploaded files.')
    
    if back_button:
        st.experimental_rerun()
else:
    st.write('Please upload both sales data and Facebook Insights data to proceed.')
