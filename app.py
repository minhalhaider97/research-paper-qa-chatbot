from src.raglearn.dataloader import ProcessPdf
from src.raglearn.vectorstore import VectorStore
from src.raglearn.search import Search


if __name__ == "__main__":
    #dataloader = ProcessPdf("data")
    #docs = dataloader.process_all_pdf()

    ## Initializing Vector Store only for once
    vectordb = VectorStore()
    #vectordb.initialize_store()
    #vectordb.add_document()

    # Now i am commenting all the Vector Store initializing things because we only run it for once

    # Search Query

    chatbot = Search()
    query = "why Traditional Network Intrusion Detection Systems is not sufficient?"
    response = chatbot.ask_query(query=query)

    print(response["question"], response["summary"])


