Towards Personalized Conversational LLMs
Ben Shi and Rajiv Swamy


Usage:

Create a conda environment with the following command:
`conda env create -f env.yml`

Create a .env file in the personal-agent directory with the following keys:
- OPENAI_API_KEY
- PINECONE_API_KEY
- PINECONE_ENVIRONMENT


To run the web app from the terminal:
- Ensure that a .env file exists in the personal-agent directory with the above keys
- activate the personal_agent conda environment with `conda activate personal_agent`
- run `streamlit run app.py`
