import streamlit as st

try:
    if 'appSecrets' in st.secrets:
        # print(st.secrets['appSecrets'])
        password = st.secrets['appSecrets']['password']
        # print(password)
except Exception as e:
    st.error(f"Failed to authenticate with Earth Engine: {e}")
    st.stop()
# Dummy users dictionary
USERS = {
    # "admin": "password123",
    "user": password
}

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in USERS and USERS[username] == password:
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

        return False
    return True
