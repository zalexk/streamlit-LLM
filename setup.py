import os

# Detect the existance of the OpenAI api Key
if os.path.exists(".streamlit/secrets.toml") == False:
    key = input("OpenAI API Key: ")
    print("---\nInput a login password to prevent unanticipated access")
    
    password = input("Password: ")
    print("---\n(Opinional) If you need to use a proxy or service emulator only")
    
    serpapi_api = input("Serpapi API Key for online search: ")
    
    base_url = input("Base_url (Press enter when unneccessary):").rstrip()
    
    if os.path.exists(".streamlit") == False: # Detect the existance of the folder
        os.makedirs(".streamlit")
        
    with open(".streamlit/secrets.toml","w") as keyfile:
        keyfile.write(f'openai_api = "{key}"')
        keyfile.write(f'\nlogin = "{password}"')
        keyfile.write(f'\nSerpapi_API_Key = "{serpapi_api}"')
        keyfile.write(f'\nbase_url = "{base_url}"')
        print("All the keys are successfully configured.")
        
else:
    print("All the keys are successfully configured.")