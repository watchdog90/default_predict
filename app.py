import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from streamlit import runtime
from streamlit.web import cli as stcli
import lightgbm
import sys
import joblib
import requests
import io



# --- section 2 ML model -------------------------------------------

def web_app():

    # --- section 2.1 Predict Diabetes From Digital Medical Records -------------------------------------------
    st.title('High Potential Default Customer Prediction APP🏆🚀')
    st.sidebar.title('AI for Default Prediction🏆🚀')
    st.subheader('⭐️ Test dataset')


    @st.cache_data(persist=True)
    def load_data():
        # Use raw GitHub URL (for public repos)
        github_url = "https://raw.githubusercontent.com/watchdog90/default_predict/main/streamtest.csv"
        data = pd.read_csv(github_url)
        # data = pd.read_csv('/app/default_predict/streamtest.csv')
        return data
    

    df = load_data()

    # (part1) ---- show test data set and correlation -----------------------------------------------------
    if st.sidebar.checkbox('Show test data', False):
        st.write(df.head(3))

    # get input data from user
    st.subheader('⭐️ Default Customer prediction')

    # load model
    # Correct URL format:
    model_url = "https://raw.githubusercontent.com/watchdog90/default_predict/main/trained_model.sav"
    @st.cache_resource
    def load_model():
        try:
            # For .pkl or .sav files:
            response = requests.get(model_url)
            model = joblib.load(io.BytesIO(response.content))
            
            # Alternative for pickle files:
            # model = pd.read_pickle(io.BytesIO(response.content))
            
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    loaded_model = load_model()
    if loaded_model:
        st.success("Model loaded successfully!")


    # Code for prediction
    submission = ''
    # creating a button for preidction
    if st.button('Make Prediction'):
        x_test = df

        y_pred = loaded_model.predict_proba(x_test)

        # Extract probabilities for class 1 (default)
        prob_default = y_pred[:, 1]  # Shape: (5000,)

        # Get user_ids from the test set
        user_ids = x_test['user_id'].values  # Shape: (5000,)

        # Combine into a DataFrame
        results = pd.DataFrame({
            'user_id': user_ids,
            'prob_default': prob_default
        })

        st.subheader('Prediction of High Potential Default Customer:')
        st.write(results)
        st.balloons()

    if st.sidebar.checkbox('Find top 10 High Potential Default Customer', False):
        x_test = df
        y_pred = loaded_model.predict_proba(x_test)

        # Extract probabilities for class 1 (efault)
        prob_default = y_pred[:, 1]  # Shape: (5000,)

        # Get user_ids from the test set
        user_ids = x_test['user_id'].values  # Shape: (5000,)

        # Combine into a DataFrame
        results = pd.DataFrame({
            'user_id': user_ids,
            'prob_default': prob_default
        })

        # Sort by probability (descending)
        ranked_results = results.sort_values(by='prob_default', ascending=False)

        # Add a rank column (1 = risk)
        ranked_results['rank'] = np.arange(1, len(ranked_results) + 1)

        # Reset index for cleaner output
        ranked_results.reset_index(drop=True, inplace=True)

        st.subheader('Top 10 High Potential Default Customer:')
        st.write(ranked_results.head(10))




if __name__ == '__main__':
    if runtime.exists():
        web_app()
    else:
        sys.argv = ['streamlit', 'run', sys.argv[0]]
        sys.exit(stcli.main())



