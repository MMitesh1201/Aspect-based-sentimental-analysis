import gradio as gr
import csv
import os
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.prompts import ChatPromptTemplate 
from langchain.output_parsers import CommaSeparatedListOutputParser 
from langchain.chains import LLMChain 
from collections import Counter
import pandas as pd
import tempfile
import io

def initialize_llm(api_key):
    """Initialize the language model with the provided API key"""
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def create_chains(llm):
    """Create the necessary chain objects"""
    aspect_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that analyzes and extracts product aspects from reviews."),
        ("human", "analyze the review and extract all the aspect of the main product in the list format and show it without any other text and in single words.")
    ])

    sentiment_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that analyzes sentiment of product's aspects in reviews."),
        ("human", """Analyze the sentiment (positive, negative, or neutral) associated with each aspect in this review: {review}
         Provide the result in this exact format, one aspect per line:
         [Aspect]: Sentiment: [Positive/Negative/Neutral]""")
    ])

    aspect_chain = LLMChain(llm=llm, prompt=aspect_extraction_prompt, output_parser=CommaSeparatedListOutputParser())
    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_analysis_prompt)
    
    return aspect_chain, sentiment_chain

def read_file_preview(file_path, file_type='csv', max_rows=10):
    """Read and format file preview"""
    if file_path is None:
        return "No file generated yet"
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path,sep=',',encoding='utf-8')
            return df.head(max_rows).to_html(classes=['table'], index=False)
        else:  # text file
            with open(file_path, 'r') as f:
                return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def process_csv_file(api_key, file):
    """Process the CSV file with reviews"""
    try:
        # Initialize LLM with API key
        llm = initialize_llm(api_key)
        aspect_chain, sentiment_chain = create_chains(llm)
        
        # Create temporary files for outputs
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'product_reviews_analyzed.csv')
        top_aspects_file = os.path.join(temp_dir, 'top_aspects.csv')
        improvements_file = os.path.join(temp_dir, 'improvements.txt')
        
        # Read the uploaded file
        df = pd.read_csv(file.name)
        
        if 'reviews.text' not in df.columns:
            return "Error: CSV file must contain a 'reviews.text' column", None, None, None
        
        # Process each review
        results = []
        for review in tqdm(df['reviews.text'].fillna(''), desc="Processing reviews"):
            try:
                # Extract aspects
                aspects = aspect_chain.run(review=review)
                
                # Analyze sentiment for each aspect
                analysis = []
                for aspect in aspects:
                    sentiment = sentiment_chain.run(review=review, aspect=aspect)
                    analysis.append((aspect, sentiment))
                
                aspects_sentiments = '; '.join([f"{aspect}: {sentiment}" for aspect, sentiment in analysis])
                results.append(aspects_sentiments)
            except Exception as e:
                results.append(f"Error processing review: {str(e)}")
        
        # Add results to dataframe
        df['Aspects_Sentiments'] = results
        
        # Save processed reviews
        df.to_csv(output_file, index=False)
        
        # Generate top aspects
        top_aspects_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that analyzes the frequency and sentiment of product aspects in reviews."),
            ("human", """Given the following list of aspects and their sentiments from product reviews:
            {aspects}
            Please analyze this data and provide:
            1. The top 5 most frequently mentioned positive aspects
            2. The top 5 most frequently mentioned negative aspects
             
            Format your response EXACTLY as follows (including the header row):
            Rank,Aspect,Sentiment,Count
            1,aspect1,Positive,42
            2,aspect2,Positive,35
            etc...
        
            Important:
          - Use exactly these column names: Type,Aspect,Sentiment,Count
          - Each row should have exactly 4 fields
          - Use only numbers in the Count column
          - Each field should not contain commas
         """)
        ])
        
        topchain = LLMChain(llm=llm, prompt=top_aspects_prompt)
        top_aspects = topchain.run({"aspects": '\n'.join(results)})
        
        with open(top_aspects_file, 'w') as f:
            f.write(top_aspects)
        
        # Generate improvement suggestions
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a product improvement specialist who analyzes customer feedback and suggests enhancements."),
            ("human", """Based on the following top positive and negative aspects of a product:
            {aspects}
            Please suggest 5 key improvements that could enhance customer satisfaction. For each suggestion:
            1. Identify the aspect you're addressing
            2. Explain the current issue or opportunity
            3. Provide a specific, actionable improvement
            4. Describe the potential impact on customer satisfaction
            Format your response as a numbered list, with each suggestion clearly separated.""")
        ])
        
        improchain = LLMChain(llm=llm, prompt=improvement_prompt)
        improvements = improchain.run({"aspects": top_aspects})
        
        with open(improvements_file, 'w') as f:
            f.write(improvements)
        
        return "Processing complete!", output_file, top_aspects_file, improvements_file
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def create_gradio_interface():
    with gr.Blocks(title="Product Review Analysis", theme=gr.themes.Default()) as app:
        gr.Markdown("""
        #           🔍 Product Review Analysis Tool
        
        ### Instructions:
        1. Enter your Google API key
        2. Upload your CSV file containing product reviews (must have a 'reviews.text' column)
        3. Click 'Process Reviews' to start the analysis
        4. View the previews and download the results
        """,)
        
        with gr.Row():
            api_key_input = gr.Textbox(
                label="Google API Key",
                placeholder="Enter your Google API key here",
                type="password",
                scale=2
            )
        
        with gr.Row():
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                scale=2
            )
        
        with gr.Row():
            process_button = gr.Button("🚀 Process Reviews", variant="primary", scale=1)
            
        with gr.Row():
            status_output = gr.Textbox(
                label="Processing Status",
                interactive=False
            )
            
        with gr.Accordion("Download Results", open=False):
            with gr.Row():
                with gr.Column():
                    analyzed_reviews = gr.File(
                        label="📊 Analyzed Reviews (CSV)"
                    )
                with gr.Column():
                    top_aspects = gr.File(
                        label="📈 Top Aspects Analysis (CSV)"
                    )
                with gr.Column():
                    improvements = gr.File(
                        label="💡 Improvement Suggestions (TXT)"
                    )
        
        with gr.Accordion("Results Preview", open=True):
            with gr.Tabs():
                with gr.TabItem("📊 Analyzed Reviews"):
                    analyzed_preview = gr.HTML()
                
                with gr.TabItem("📈 Top Aspects"):
                    top_aspects_preview = gr.HTML()
                
                with gr.TabItem("💡 Improvement Suggestions"):
                    improvements_preview = gr.HTML()
        
        def update_previews(status, analyzed_file, aspects_file, improvements_file):
            if status.startswith("Error"):
                return (
                    "No preview available",
                    "No preview available",
                    "No preview available"
                )
            
            analyzed_html = read_file_preview(analyzed_file, 'csv')
            aspects_html = read_file_preview(aspects_file, 'csv')
            improvements_text = read_file_preview(improvements_file, 'text')
            
            # Add some styling to the previews
            analyzed_html = f"""
            <div style="max-height: 400px; overflow-y: auto;">
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #f5f5f5; }}
                    tr:nth-child(even) {{ background-color: #242222; }}
                </style>
                {analyzed_html}
            </div>
            """
            
            aspects_html = f"""
            <div style="max-height: 400px; overflow-y: auto;">
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #f5f5f5; }}
                    tr:nth-child(even) {{ background-color: #242222; }}
                </style>
                {aspects_html}
            </div>
            """
            
            improvements_html = f"""
            <div style="max-height: 400px; overflow-y: auto;">
                <pre style="white-space: pre-wrap; padding: 15px; background-color: #242222; border-radius: 5px;">
                {improvements_text}
                </pre>
            </div>
            """
            
            return analyzed_html, aspects_html, improvements_html

        # Set up the processing chain
        process_button.click(
            fn=process_csv_file,
            inputs=[api_key_input, file_input],
            outputs=[status_output, analyzed_reviews, top_aspects, improvements]
        ).then(
            fn=update_previews,
            inputs=[status_output, analyzed_reviews, top_aspects, improvements],
            outputs=[analyzed_preview, top_aspects_preview, improvements_preview]
        )
        
        gr.Markdown("""
        ### 📝 Output Files Description:
        1. **Analyzed Reviews**: Contains your original reviews with added aspect and sentiment analysis for each review
        2. **Top Aspects**: Lists the most frequently mentioned positive and negative aspects across all reviews
        3. **Improvement Suggestions**: Provides actionable recommendations based on the analysis results
        
        ### ℹ️ Note:
        - Make sure your CSV file has a column named 'reviews.text'
        - Processing time depends on the number of reviews
        - Keep your API key secure and never share it
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
