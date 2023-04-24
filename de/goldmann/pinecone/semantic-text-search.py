import torch
from sentence_transformers import SentenceTransformer
import pinecone
from decouple import config

pinecone.init(
    api_key=config('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=config('PINECONE_ENVIRONMENT') # next to api key in console
)

# now connect to the index
index = pinecone.GRPCIndex('keyword-search')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def firstQuery():
    #query = "which city has the highest population in the world?"
    query = "what is 'Identify inefficiencies'"

    # create the query vector
    xq = model.encode(query).tolist()

    # now query
    xc = index.query(xq, top_k=3, include_metadata=True)

    for result in xc['matches']:
        print(f"{round(result['score'], 2)}: {result['metadata']['text']}")


if __name__ == '__main__':
    firstQuery()