import json
import pinecone
import docx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from decouple import config

index_name = 'keyword-search'

def generateJson(pathToDocs):
    all_sentences = []
    doc = docx.Document(pathToDocs)
    for paragraph in doc.paragraphs:
        all_sentences.append(paragraph.text)

    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

    all_embeddings = model.encode(all_sentences)

    # transfo-xl tokenizer uses word-level encodings
    tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')

    all_tokens = [tokenizer.tokenize(sentence.lower()) for sentence in all_sentences]

    upserts = {'vectors': []}
    for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
        vector = {'id':f'{i}',
                  'values': embedding.tolist(),
                  'metadata':{'tokens':tokens}}
        upserts['vectors'].append(vector)

    #print(upserts)
    #print(len(upserts['vectors'][0]['values']))

    pinecone.init(
        api_key=config('PINECONE_API_KEY'),  # find at app.pinecone.io
        environment=config('PINECONE_ENVIRONMENT') # next to api key in console
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=len(upserts['vectors'][0]['values']))

    with open('./upsert.json', 'w') as f:
        json.dump(upserts, f, indent=4)


if __name__ == '__main__':
    pathToWordFile = "C:\\Users"
    #pathToWordFile ="C:\\Users\\agol\\github\\pinecone-testing\\data"
    generateJson(pathToWordFile + "\\Strategic_Alignment_Process.docx")