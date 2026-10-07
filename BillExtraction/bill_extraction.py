from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
import json
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


def get_pdf_data(fpath):
    text = ""

    pdf = PdfReader(fpath)

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


def get_llm():
    client = Groq(
        api_key=api_key
    )

    return client


def get_invoice_info_from_llm(data):

    llm = get_llm()

    prompt = """
Act as an expert in extracting information from medical invoices.

You are given the invoice details of a patient.

Go through the document carefully and extract:

1. Disease
2. Expense amount

Return ONLY valid JSON in the following format:

{
    "disease": "",
    "expense": ""
}
"""

    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": f"INVOICE DETAILS:\n{data}"
        }
    ]

    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.4,
        max_tokens=500
    )

    response_text = response.choices[0].message.content

    invoice_data = json.loads(response_text)

    return invoice_data


if __name__ == '__main__':

    bill_folder = "Bills"
    bill_name = "MedicalBill1.pdf"

    bill_path = os.path.join(bill_folder, bill_name)

    if not os.path.exists(bill_path):

        print(f"{bill_path} does not exist. Please check the file location")

    else:

        bill_info = get_pdf_data(bill_path)

        invoice_details = get_invoice_info_from_llm(bill_info)

        print(f"Disease: {invoice_details['disease']}")
        print(f"Expense: {invoice_details['expense']}")