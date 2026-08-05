#streamlit run main.py &> log.txt & <---- run in terminal first
from pyngrok import ngrok
ngrok.set_auth_token("3HGkO6mPOrf8wMeYwz8aUD1g3AG_5G9BNL6UWRCXU8ZL7yu6Y")
public_url = ngrok.connect(8501)
print(public_url)