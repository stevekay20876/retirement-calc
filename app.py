import streamlit as st

# 1. Set Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Coming Soon | Under Construction", 
    page_icon="🚧", 
    layout="centered"
)

# 2. Hide Streamlit's default header, footer, and hamburger menu for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# 3. Main Content
st.markdown("<h1 style='text-align: center;'>🚧 We are under construction! 🚧</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>Something awesome is in the works.</h3>", unsafe_allow_html=True)

st.write("")
st.write("<p style='text-align: center;'>We are currently revamping our website to bring you a better experience. We will be launching very soon!</p>", unsafe_allow_html=True)
st.write("")

# 4. Mock Progress Bar
progress_text = "Development in progress... 75% complete"
my_bar = st.progress(75, text=progress_text)

st.divider()

# 5. Email Capture Form
st.write("### Get Notified")
st.write("Leave your email below and we'll let you know the second we go live.")

with st.form("notify_form", clear_on_submit=True):
    email = st.text_input("Enter your email address:", placeholder="hello@example.com")
    submitted = st.form_submit_button("Notify Me 🚀")
    
    if submitted:
        if email and "@" in email:
            # Here you would typically connect to a database or API (like Mailchimp)
            st.success("Thanks! We'll keep you posted.")
        else:
            st.error("Please enter a valid email address.")

st.divider()

# 6. Footer / Social Links
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Contact us: <a href="mailto:info@yourwebsite.com">info@yourwebsite.com</a></p>
        <p>Follow us on <a href="https://twitter.com">Twitter</a> | <a href="https://linkedin.com">LinkedIn</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)