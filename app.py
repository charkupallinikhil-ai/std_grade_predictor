import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    return None

model = load_model()

# Load Data for Dashboard
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('student_dataset.xlsx')
        df['StudyHours'] = pd.to_numeric(df['StudyHours'], errors='coerce')
        return df
    except:
        return None

df = load_data()

st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Select a Module:", ["Dashboard", "Single Prediction", "Batch Prediction"])

if page == "Dashboard":
    st.title("📊 Student Performance Dashboard")
    st.markdown("Explore the historical student dataset to understand the factors affecting grades.")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df))
        col2.metric("Avg Study Hours", f"{df['StudyHours'].mean():.2f}")
        col3.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
        col4.metric("Avg Midterm Score", f"{df['Midterm'].mean():.1f}")
        
        st.markdown("---")
        
        col_charts1, col_charts2 = st.columns(2)
        
        with col_charts1:
            st.subheader("Grade Distribution")
            grade_counts = df['Grade'].value_counts().reset_index()
            grade_counts.columns = ['Grade', 'Count']
            # Sort grades logically if they are letters A, B, C, D, F
            grade_counts = grade_counts.sort_values(by='Grade')
            fig_grade = px.bar(grade_counts, x='Grade', y='Count', color='Grade', 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_grade, use_container_width=True)
            
        with col_charts2:
            st.subheader("Study Hours vs. Midterm Score")
            fig_scatter = px.scatter(df, x='StudyHours', y='Midterm', color='Result',
                                     hover_data=['Grade'], color_discrete_sequence=['#ef553b', '#00cc96'])
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Feature Correlation")
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
            
    else:
        st.error("Dataset not found. Please ensure 'student_dataset.xlsx' is in the directory.")

elif page == "Single Prediction":
    st.title("🎯 Predict Student Grade")
    st.markdown("Enter the student's details below to predict their final grade.")
    
    if model is None:
        st.error("Model not found. Please train the model first by running `train_model.py`.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=15, max_value=60, value=20)
                study_hours = st.slider("Study Hours (per week)", min_value=0.0, max_value=40.0, value=10.0, step=0.5)
                attendance = st.slider("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
                
            with col2:
                assignments = st.slider("Assignments Score (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                midterm = st.slider("Midterm Exam Score", min_value=0, max_value=100, value=70)
            
            submitted = st.form_submit_button("Predict Grade")
            
            if submitted:
                input_data = pd.DataFrame({
                    'Age': [age],
                    'StudyHours': [study_hours],
                    'Attendance': [attendance],
                    'Assignments': [assignments],
                    'Midterm': [midterm]
                })
                
                prediction = model.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                # Assign colors based on grade
                color_map = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': '#d4a017', 'F': 'red'}
                pred_color = color_map.get(prediction, 'gray')
                
                st.markdown(f"<h1 style='text-align: center; color: {pred_color};'>Predicted Grade: {prediction}</h1>", unsafe_allow_html=True)
                
                # Show probabilities if model supports it (Random Forest does)
                try:
                    proba = model.predict_proba(input_data)[0]
                    classes = model.classes_
                    proba_df = pd.DataFrame({'Grade': classes, 'Probability': proba})
                    proba_df = proba_df.sort_values(by='Probability', ascending=False)
                    
                    st.markdown("#### Prediction Confidence")
                    fig_proba = px.bar(proba_df, x='Probability', y='Grade', orientation='h', 
                                       text=[f"{p:.1%}" for p in proba_df['Probability']],
                                       color='Probability', color_continuous_scale='Blues')
                    fig_proba.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_proba, use_container_width=True)
                except AttributeError:
                    pass

elif page == "Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV or Excel file containing student data to get predictions for multiple students at once.")
    st.markdown("**Required columns:** `Age`, `StudyHours`, `Attendance`, `Assignments`, `Midterm`")
    
    if model is None:
        st.error("Model not found. Please train the model first by running `train_model.py`.")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)
                    
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head())
                
                # Check for required columns
                required_cols = ['Age', 'StudyHours', 'Attendance', 'Assignments', 'Midterm']
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("Generate Predictions"):
                        with st.spinner("Predicting..."):
                            # Handle 'unknown' in StudyHours if exists
                            if input_df['StudyHours'].dtype == object:
                                input_df['StudyHours'] = pd.to_numeric(input_df['StudyHours'], errors='coerce')
                                
                            predictions = model.predict(input_df[required_cols])
                            input_df['Predicted_Grade'] = predictions
                            
                            st.success("Predictions generated successfully!")
                            st.dataframe(input_df.head(10))
                            
                            # Convert df to csv for download
                            csv = input_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name='student_grade_predictions.csv',
                                mime='text/csv',
                            )
            except Exception as e:
                st.error(f"Error processing file: {e}")
