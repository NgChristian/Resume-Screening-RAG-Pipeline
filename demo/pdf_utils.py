from pypdf import PdfReader
import pandas as pd
from typing import List, Union, BinaryIO
import re
import glob

def extract_text_from_pdf(pdf_file: Union[str, BinaryIO]) -> str:
    """
    Extracts raw text from a PDF file.
    
    Args:
        pdf_file: Path to the PDF file or file object
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def process_pdf_files(pdf_files: List[Union[str, BinaryIO]]) -> str:
    """
    Processes a list of PDF files and returns a CSV with the extracted texts.
    
    Args:
        pdf_files: List of PDF files (paths or file objects)
        
    Returns:
        str: In-memory CSV data with ID and Resume columns
    """
    cv_data = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            # Extract raw text
            text = extract_text_from_pdf(pdf_file)
            
            # Add to CV list
            if text:  # Skip empty files
                cv_data.append({
                    'ID': idx,  # Sequential numbering starting from 1
                    'Resume': text
                })
                
        except Exception as e:
            file_name = getattr(pdf_file, 'name', str(pdf_file))
            print(f"Error with file {file_name}: {str(e)}")
    
    # Create DataFrame and convert to CSV
    if cv_data:
        df = pd.DataFrame(cv_data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    return ""