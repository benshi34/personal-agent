import openai
import pinecone
from prompts import INFO_AGENT_PROMPT
import os
import time
import json
import itertools
from uuid import uuid4
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

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
    def __init__(self, user):

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
    def read(self, text, k = None):
        
        # If k is None, then dynamically determin the k nearest neighbors to
        # include in contest

        matches = None

        if k is None:
            matches = self._index_query(text, k=100, include_values=False)['matches']
            
            # should be sorted in descending order
            scores = [match['score'] for match in matches] 

            indexes_keep = self._determine_indices_kde(scores)

            matches = [matches[i] for i in indexes_keep]
        
        else:
            matches = self._index_query(text, k=k, include_values=False)['matches']
            
        
        # create context given a set of matches
        context = ""

        for match in matches:
            context += match['metadata']['text'] + '\n'

        return context
    
    # similar to read, but provides the matches list
    # can access scores through match['score']
    # can access text through match['metadata']['text']
    def _index_query(self, text, k, include_values):
        query_embedding = self._get_text_embedding(text)

        res = self.index.query(
                                [query_embedding], 
                                top_k=k, 
                                include_metadata=True,
                                include_values=include_values
                            )

        return res
    
    # use manual cosine similarity cutoff to determine items
    # include in context
    def _determine_indices_manual(self, scores, cutoff):

        indices = np.array(range(len(scores)))
        scores = np.array(scores)

        ret = indices[scores > cutoff].tolist()

        if len(ret) == 0:
            return [1]
        
        return ret
    
    # Use kmeans with self-supplied value for k
    # assume k is between 2 and 5 for most data
    def _determine_indices_kmeans(self, scores, k=3):
        indices = np.array(range(len(scores)))

        X = np.array([scores]).T
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        max_label = np.argmax(kmeans.cluster_centers_)

        mask = kmeans.labels_.flatten() == max_label

        return indices[mask].tolist()
    
    # https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
    # It's not obvious what the optimal bandwidth is for KDE
    def _determine_indices_kde(self, scores):

        X = np.array([scores]).T

        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

        s = np.linspace(0,1,50)

        pdf = np.exp(kde.score_samples(s.reshape(-1,1)))

        mi = argrelextrema(pdf, np.less)[0]

        cutoff = np.max(s[mi])

        return self._determine_indices_manual(scores, cutoff)

    def write(self, text):

        # preprend text with timestamp
        timestamp = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
        text = f"{timestamp}: {text}"

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
    
    # https://docs.pinecone.io/docs/insert-data
    def _chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
    
    def batch_write(self, texts):
        vectors = []

        for text in texts:
            # preprend text with timestamp
            timestamp = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
            text = f"{timestamp}: {text}"

            embedding = self._get_text_embedding(text) # list rep of embedding

            vectors.append(
                (
                    str(uuid4()), # Vector ID
                    embedding, # Dense vector
                    {
                        'user': self.user,
                        'text': text
                    },
                )
            )
        
        for ids_vectors_chunk in self._chunks(vectors):
            self.index.upsert(vectors=ids_vectors_chunk)
    
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
# Store data locally and load from existing files
class PrimitiveDatastore:
    def __init__(self, user, file=None) -> None:
        self.dataset = []
        self.user = user

        if file is not None:
            f = open(file)
            self.dataset = json.load(f)

    def read(self):
        pass
    def write(self):
        pass

    def save_dataset(self, path):
        out_file = open(path, 'w')
        json.dump(self.dataset, out_file)