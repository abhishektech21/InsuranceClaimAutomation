import os
import re
import json

from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


# Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)


# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

general_exclusion_list = [
    "HIV/AIDS",
    "Parkinson's disease",
    "Alzheimer's disease",
    "pregnancy",
    "substance abuse",
    "self-inflicted injuries",
    "sexually transmitted diseases(std)",
    "pre-existing conditions"
]


def get_document_loader():
    loader = DirectoryLoader(
        "documents",
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader
    )

    return loader.load()


def get_text_chunks(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_documents(documents)


def get_embeddings():

    documents = get_document_loader()

    chunks = get_text_chunks(documents)

    # Local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return db


def get_retriever():

    db = get_embeddings()

    return db.as_retriever()


def get_claim_approval_context():

    db = get_embeddings()

    context = db.similarity_search(
        "What are the documents required for claim approval?"
    )

    claim_approval_context = ""

    for document in context:
        claim_approval_context += document.page_content

    return claim_approval_context


def get_general_exclusion_context():

    db = get_embeddings()

    context = db.similarity_search(
        "Give a list of all general exclusions"
    )

    general_exclusion_context = ""

    for document in context:
        general_exclusion_context += document.page_content

    return general_exclusion_context


def get_file_content(file):

    text = ""

    if file.filename.endswith(".pdf"):

        pdf = PdfReader(file)

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


def get_bill_info(data):

    prompt = """
Act as an expert in extracting information from medical invoices.

Extract:

1. Disease
2. Expense amount

Return ONLY valid JSON in this format:

{
    "disease": "",
    "expense": ""
}

If information is unavailable, use null.
"""

    response = llm.invoke(
        prompt + f"\n\nINVOICE DETAILS:\n{data}"
    )

    content = response.content.strip()

    # Remove markdown code blocks if generated
    content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)


PROMPT = """
You are an AI assistant for verifying health insurance claims.

You are given references for approving the claim and patient details.

Analyse the given data and predict whether the claim should be accepted or rejected.

Follow these guidelines:

1. Verify whether the patient has provided all necessary information and documents.

If information or documents are incomplete:

INFORMATION = FALSE

Reject the claim.

If all required information and documents are provided:

INFORMATION = TRUE


2. Check whether any disease mentioned in the medical bill is present in the general exclusions list.

If present:

EXCLUSION = FALSE

Reject the claim.


DOCUMENTS FOR CLAIM APPROVAL:

{claim_approval_context}


GENERAL EXCLUSION LIST:

{general_exclusion_context}


PATIENT INFO:

{patient_info}


MEDICAL BILL:

{medical_bill_info}


Maximum claim amount:

{max_amount}


Generate a detailed report.

Write whether INFORMATION and EXCLUSION are TRUE or FALSE.

Reject the claim if any criteria is FALSE.

Write whether the claim is accepted or rejected.

If accepted, mention the maximum amount which can be approved.

Report format:

Executive Summary

Introduction

Claim Details

Claim Description

Document Verification

Document Summary

Fraud Verification
"""


def check_claim_rejection(
    claim_reason,
    general_exclusion_list,
    prompt_template,
    threshold=0.4
):

    if not claim_reason:
        return prompt_template

    vectorizer = CountVectorizer()

    patient_info_vector = vectorizer.fit_transform(
        [claim_reason]
    )

    for disease in general_exclusion_list:

        disease_vector = vectorizer.transform(
            [disease]
        )

        similarity = cosine_similarity(
            patient_info_vector,
            disease_vector
        )[0][0]

        if float(similarity) > float(threshold):

            prompt_template = """
You are an AI assistant for verifying health insurance claims.

PATIENT INFO:

{patient_info}


CLAIM MUST BE REJECTED.

The patient has {disease}, which is present in the general exclusion list.


Executive Summary

Introduction

Claim Details

Claim Description

Document Verification

Document Summary
"""

            return prompt_template

    return prompt_template


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/', methods=['POST'])
def msg():

    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']

    medical_bill = request.files['medical_bill']

    total_claim_amount = request.form['total_claim_amount']

    description = request.form['description']


    # Extract text from uploaded bill
    bill = get_file_content(medical_bill)


    # Extract disease and expense using Groq
    bill_info = get_bill_info(bill)


    # Reject if claimed amount is greater than bill amount
    if (
        bill_info.get("expense") is not None
        and int(bill_info["expense"].replace(",", "")) < int(total_claim_amount)
    ):

        output = (
            "The amount mentioned for claiming is more than "
            "the billed amount. Claim Rejected."
        )

        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=output
        )


    elif bill_info.get("expense") is not None:

        patient_info = (
            f"Name: {name}\n"
            f"Address: {address}\n"
            f"Claim Type: {claim_type}\n"
            f"Claim Reason: {claim_reason}\n"
            f"Medical Facility: {medical_facility}\n"
            f"Date: {date}\n"
            f"Total Claim Amount: {total_claim_amount}\n"
            f"Description: {description}"
        )


        medical_bill_info = f"Medical Bill:\n{bill}"


        validated_prompt = check_claim_rejection(
            bill_info.get("disease"),
            general_exclusion_list,
            PROMPT
        )


        prompt_template = PromptTemplate(
            input_variables=[
                "claim_approval_context",
                "general_exclusion_context",
                "patient_info",
                "medical_bill_info",
                "max_amount",
                "disease"
            ],
            template=validated_prompt
        )


        formatted_prompt = prompt_template.format(
            claim_approval_context=get_claim_approval_context(),
            general_exclusion_context=get_general_exclusion_context(),
            patient_info=patient_info,
            medical_bill_info=medical_bill_info,
            max_amount=total_claim_amount,
            disease=bill_info.get("disease")
        )


        # Generate final report using Groq
        response = llm.invoke(
            formatted_prompt
        )

        output = response.content

        output = re.sub(
            r'\n',
            '<br>',
            output
        )


        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=output
        )


    else:

        output = "Please enter a valid Consultation Receipt."

        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=output
        )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8081,
        debug=True
    )