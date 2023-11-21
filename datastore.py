import openai
import pinecone
from prompts import INFO_AGENT_PROMPT
import os
import time
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(override=True)

openai.api_key = os.environ.get('OPENAI_API_KEY')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

EMB_MODEL = "text-embedding-ada-002"

INDEX_NAME = 'personal-agent'

EMBED_DIM = 1536 # text embedding vector size

class PineconeDatastore:
    def __init__(self, user='Alice'):

        self.user = user

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

        if INDEX_NAME not in pinecone.list_indexes():
            print(f"Creating {INDEX_NAME} index from scratch")
            pinecone.create_index(
                name=INDEX_NAME, 
                dimension=EMBED_DIM,
                metric='cosine'
            )

            # wait for index to be fully initialized
            time.sleep(1)
        
        self.index = pinecone.Index(INDEX_NAME)

        print("Pinecone Datastore initialized!")

    # gets the top-k matches and organizes them into one string
    # no dynamic 
    def read(self, text, k = 1):

        matches = self.read_analysis(text, k)

        context = ""

        for match in matches:
            context += match['metadata']['text'] + '\n'

        return context
    
    # similar to read, but provides the matches list
    # can access scores through match['score']
    # can access text through match['metadata']['text']
    def read_analysis(self, text, k = 1):
        query_embedding = self._get_text_embedding(text)

        res = self.index.query([query_embedding], top_k=k, include_metadata=True)

        return res['matches']

    def write(self, text):

        embedding = self._get_text_embedding(text) # list rep of embedding

        metadata = {
            'user': self.user,
            'text': text
        }

        self.index.upsert(
            vectors=[
                (
                    str(uuid4()), # Vector ID
                    embedding, # Dense vector
                    metadata,
                )
            ]
        )

        return "Write complete!"
    
    def _get_text_embedding(self, text):
        # Get embedding for text

        res = openai.Embedding.create(
            input=[text],
            model=EMB_MODEL
        )

        return res['data'][0]['embedding'] # list rep of embedding

    # Only call if you want to delete the entire index!
    # Can't be reversed
    def delete_index(self):
        pinecone.delete_index(INDEX_NAME)


# Class for doing primitive KNN and datastore
class PrimitiveDatastore:
    def read(self):
        pass
    def write(self):
        pass