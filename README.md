# Towards Personalized Conversational LLMs
By: Ben Shi and Rajiv Swamy


## Usage:

Create a conda environment with the following command:

```conda env create -n personal_agent -f env.yml```

Create a .env file in the personal-agent directory with the following keys:
```
OPENAI_API_KEY: <YOUR KEY HERE>
PINECONE_API_KEY: <YOUR KEY HERE>
PINECONE_ENVIRONMENT: <YOUR KEY HERE>
```

To run the web app from the terminal:
1. Ensure that a .env file exists in the personal-agent directory with the above keys
2. activate the personal_agent conda environment with the command ```conda activate personal_agent```
3. run ```streamlit run app.py```
