import os
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

class ProcessPdf:
    def __init__(self, pdf_directory):
        self.pdf_dir = Path(pdf_directory)
        self.all_documents = []

    def process_all_pdf(self):
        pdf_files = list(self.pdf_dir.glob("**/*.pdf"))
        for pdf in pdf_files:
            loader = PyMuPDFLoader(str(pdf))
            loaded_pdf = loader.load()

            for doc in loaded_pdf:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'
            self.all_documents.extend(loaded_pdf)
            print(f" Successfully Loaded {len(loaded_pdf)} pages")

        print(f"Total document loaded {len(self.all_documents)}")

        return self.all_documents
