import docx2txt
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from decouple import config

def readDocfile():
    # extract text

    return docx2txt.process("C:\\Users\\agol\\github\\pinecone-testing\\data\\Strategic_Alignment_Process.docx")

    # extract text and write images in /tmp/img_dir
    #text = docx2txt.process("file.docx", "/tmp/img_dir")

def storeContentInPinecone(content):
    #doc = docx.Document("C:\\Users\\agol\\github\\pinecone-testing\\data\\Strategic_Alignment_Process.docx")
    #result = [p.text for p in doc.paragraphs]

    #print(result)
    #print(content)
    all_sentences = [
        "purple is the best city in the forest",
        "No way chimps go bananas for snacks!",
        "it is not often you find soggy bananas on the street",
        "green should have smelled more tranquil but somehow it just tasted rotten",
        "joyce enjoyed eating pancakes with ketchup",
        "throwing bananas on to the street is not art",
        "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled",
        "I'm getting way too old. I don't even buy green bananas anymore.",
        "to get your way you must not bombard the road with yellow fruit",
        "Time flies like an arrow; fruit flies like a banana"
    ]

    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

    all_embeddings = model.encode(all_sentences)
    all_embeddings.shape

    # transfo-xl tokenizer uses word-level encodings
    tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')

    all_tokens = [tokenizer.tokenize(sentence.lower()) for sentence in all_sentences]
    all_tokens[0]

    #embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceEmbeddings()
    #model_name = "sentence-transformers/all-mpnet-base-v2"
    #model_kwargs = {'device': 'cpu'}
    #embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


    pinecone.init(
        api_key=config('PINECONE_API_KEY'),  # find at app.pinecone.io
        environment=config('PINECONE_ENVIRONMENT') # next to api key in console
    )
    if 'keyword-search' not in pinecone.list_indexes():
        pinecone.create_index(name='keyword-search', dimension=all_embeddings.shape[0])
    index = pinecone.Index('keyword-search')
    index.delete(deleteAll='true')
    #index.delete()
    #index.close()

    #upserts = [(v['id'], v['values'], v['metadata']) for v in all_embeddings]
    # then we upsert
    #upserts = {'vectors': []}
    #for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
    #    vector = {'id':f'{i}',
    #              'values': embedding.tolist(),
    #              'metadata':{'tokens':tokens}}
    #    upserts['vectors'].append(vector)
    #index.upsert(vectors=upserts)
    #with open('./upsert.json', 'w') as f:
    #    json.dump(upserts, f, indent=4)
    #upserts = {'vectors': []}
    #for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
    #    vector = {'id':f'{i}',
    #              'values': embedding.tolist(),
    #              'metadata':{'tokens':tokens}}
    #    upserts['vectors'].append(vector)

    # das ist die Suche, hier wird nix gespeichert
    #docsearch = Pinecone.from_texts(
    #    text, embeddings,
    #    index_name=index_name, namespace=namespace)

    batch_size = 64
    #index = pinecone.Index(index_name)
    # Insert sample data (5 8-dimensional vectors)
    #index.upsert([
    #    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    #    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    #    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    #    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    #    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    #])


if __name__ == '__main__':
    #args = process_args()
    #text = process(args.docx, args.img_dir)
    text_test = readDocfile()
    #sys.stdout.write(text)
    #text_test = "What are 5 vacation destinations for someone who likes to eat pasta?"
    storeContentInPinecone(text_test)