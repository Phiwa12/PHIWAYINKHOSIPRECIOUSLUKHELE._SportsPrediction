import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')  # Load the model
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Define columns based on model's input features
cols = ['movement_reactions', 'potential', 'passing', 'wage_eur', 'mentality_composure',
        'value_eur', 'dribbling', 'attacking_short_passing', 'mentality_vision', 
        'international_reputation']

def main():
    st.title("Player's Overall Rating")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Player's Overall Rating App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Input fields for each feature
    movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, value=50)
    potential = st.number_input("Potential", min_value=0, max_value=100, value=50)
    passing = st.number_input("Passing", min_value=0, max_value=100, value=50)
    wage_eur = st.number_input("Wage (EUR)", min_value=0, value=10000)
    mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, value=50)
    value_eur = st.number_input("Value (EUR)", min_value=0, value=100000)
    dribbling = st.number_input("Dribbling", min_value=0, max_value=100, value=50)
    attacking_short_passing = st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=50)
    mentality_vision = st.number_input("Mentality Vision", min_value=0, max_value=100, value=50)
    international_reputation = st.number_input("International Reputation", min_value=1, max_value=5, value=3)
    
    if st.button("Predict"):
        # Create a dictionary with the input data
        data = {
            'movement_reactions': movement_reactions,
            'potential': potential,
            'passing': passing,
            'wage_eur': wage_eur,
            'mentality_composure': mentality_composure,
            'value_eur': value_eur,
            'dribbling': dribbling,
            'attacking_short_passing': attacking_short_passing,
            'mentality_vision': mentality_vision,
            'international_reputation': international_reputation
        }
        # Create a DataFrame from the dictionary
        df = pd.DataFrame([data], columns=cols)

        # Transform the input data using the loaded scaler
        scaled_features = scaler.transform(df)
        
        # Make a prediction using the loaded model
        prediction = model.predict(scaled_features)
        
        st.write("Input features:")
        st.write(df)
        
        st.success(f"Predicted Overall Rating: {prediction[0]}")

if __name__ == '__main__':
    main()
