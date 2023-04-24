import pinecone
from decouple import config
from pinecone_datasets import load_dataset

#dataset = load_dataset('quora_all-MiniLM-L6-bm25')
dataset = load_dataset('quora', split='train')
# we drop sparse_values as they are not needed for this example
dataset.documents.drop(['sparse_values', 'metadata'], axis=1, inplace=True)
dataset.documents.rename(columns={'blob': 'metadata'}, inplace=True)
dataset.head()


pinecone.init(
    api_key=config('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=config('PINECONE_ENVIRONMENT') # next to api key in console
)

if 'keyword-search' not in pinecone.list_indexes():
    pinecone.create_index(
        name='keyword-search',
        dimension=len(dataset.documents.iloc[0]['values']),
        metric='cosine'
    )

index = pinecone.GRPCIndex('keyword-search')

index.upsert_from_dataframe(dataset.documents)