
import streamlit as st


st.title("classification page")
st.write("This is the home page of the application.")

# Example of a form on the home page
with st.form(key="home_form"):
    username = st.text_input("Username")
    password = st.text_input("Password")
    submit_button = st.form_submit_button("Login")

    if submit_button:
        st.write(f"Welcome, {username}!")