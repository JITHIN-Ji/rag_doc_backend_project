import openai
from app.services.rag_pipeline import RAGPipeline
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse,  Message
import logging
import os
from langdetect import detect  
import pycountry
from typing import List, Dict
from groq import AsyncGroq
from app.models import question_store  
import uuid 

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, user_id: str):
        
        self.rag_pipeline = RAGPipeline(user_id)
        self.openai_client = self.rag_pipeline.openai_client
        # self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY")) 
        logger.info("AgentService initialized.")

    async def handle_document_upload(self, file_path: str, document_id: str) -> DocumentUploadResponse:
        """
        Agent decides to process and embed the uploaded document.
        """
        logger.info(f"Agent handling document upload: {document_id} at {file_path}")

        
        
        logger.info("Cleared previous FAISS index and metadata.")

        
        success, message = await self.rag_pipeline.process_and_embed_document(file_path, document_id)

        if not success:
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up failed upload: {file_path}")
            except Exception:
                logger.warning(f"Failed to clean up upload file {file_path} after error: {message}")
            return DocumentUploadResponse(message=message, document_id=document_id, filename=document_id)

        final_message = f"Successfully processed '{document_id}'. You can now ask questions."
        return DocumentUploadResponse(message=final_message, document_id=document_id, filename=document_id, success=True)
    
    async def handle_image_upload(self, extracted_text: str, document_id: str) -> DocumentUploadResponse:
        logger.info(f"Agent handling image upload for: {document_id}")

        
        
        logger.info("Cleared previous FAISS index and metadata.")

        
        pages = [{"index": 0, "label": "1", "text": extracted_text}]

        
        chunks, meta_per_chunk = self.rag_pipeline.doc_processor.chunk_text(pages)
        if not chunks:
            msg = f"No content found in OCR for {document_id}"
            logger.warning(msg)
            return DocumentUploadResponse(message=msg, document_id=document_id, filename=document_id)

        await self.rag_pipeline.process_and_embed_document_from_chunks(
            chunks, meta_per_chunk, document_id
        )

        final_message = f"Successfully processed image '{document_id}'. You can now ask questions."
        return DocumentUploadResponse(message=final_message, document_id=document_id, filename=document_id, success=True)
    
    async def handle_url_ingestion(self, url: str) -> DocumentUploadResponse:
        """
        Handles the ingestion of a web page from a URL.
        """
        # Generate a unique ID for this web content, similar to a file.
        document_id = str(uuid.uuid4())
        logger.info(f"Handling URL ingestion for {url} with new doc ID: {document_id}")

        # Call the new method in the RAG pipeline to scrape, chunk, and embed.
        success, message = await self.rag_pipeline.process_and_embed_web_content(
            url=url,
            document_id=document_id
        )

        # Return a consistent response object that the frontend can handle.
        final_message = f"Content successfully ingested from '{url}'. You can now ask questions."
        return DocumentUploadResponse(message=final_message, document_id=document_id, filename=url, success=True)


    async def _rephrase_with_openai(
        self,
        history: List[Dict[str, str]],
        latest_query: str,
        model: str = "openai/gpt-4o-mini"
    ) -> str:
        """Rewrite a follow-up question using OpenAI only if it's related to previous queries."""
        if not history:
            return latest_query

        recent_queries = history[-5:] if len(history) > 5 else history
        history_str = "\n".join([f"- {h['query']}" for h in reversed(recent_queries)])
        
        logger.info(f"Rephrasing latest query '{latest_query}' with OpenAI model '{model}' using history:\n{history_str}")
        
        prompt = f"""You are an expert conversational assistant. Your goal is to understand a user's intent from a chat history and rewrite their last message into a perfect, standalone question.

Conversation History (most recent first):
{history_str}

User's Last Message: "{latest_query}"

**Your Task:**

1.  **Analyze:** Read the history and the user's last message. Does the last message depend on the history to make sense? (e.g., using "it", "them", or asking a short follow-up).

2.  **Rewrite if Necessary:** If the message is a follow-up, rewrite it into a full question by adding the missing context from the history. Make it sound natural.

3.  **Do Nothing if Complete:** If the user's last message is already a complete, standalone question, you MUST respond with the exact text: "NO_REPHRASE_NEEDED".

**Examples:**
-   **History:** "What is Selenium?" -> **Last Message:** "list the tools?"
    **Your Output:** "What are the tools in the Selenium suite?"

-   **History:** "Tell me about the SafePulse report." -> **Last Message:** "what are its main findings?"
    **Your Output:** "What are the main findings of the SafePulse report?"

-   **History:** (any) -> **Last Message:** "What is the capital of Germany?"
    **Your Output:** "NO_REPHRASE_NEEDED"

Respond with ONLY the rewritten query OR the text "NO_REPHRASE_NEEDED".
"""
        try:
            resp = await self.openai_client.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
            )
            
            result = resp.choices[0].message.content.strip()
            
            if "NO_REPHRASE_NEEDED" in result:
                logger.info("No rephrasing needed - query is standalone or unrelated")
                return latest_query
            else:
                logger.info(f"Rephrased query: '{result}'")
                return result
                
        except Exception as e:
            logger.error(f"OpenAI rephrase failed: {e}")
            return latest_query

    async def handle_chat_query(self, request: ChatRequest, user_id: str) -> ChatResponse:
        """
        Process a chat query by orchestrating RAG steps for multilingual support.
        """
        logger.info(f"Agent handling chat query for user {user_id}: {request.query}")

        # Step 1: Get chat history and create a standalone query
        chat_history_list = await question_store.get_recent(user_id)
        standalone_query = await self._rephrase_with_openai(
            history=chat_history_list,
            latest_query=request.query,
        )

        # Step 2: Detect the user's desired language
        language_code = request.language or "auto"
        if language_code == "auto":
            try:
                language_code = detect(request.query)
                logger.info(f"Detected language: {language_code}")
            except Exception:
                language_code = "en"
        try:
            language_name = pycountry.languages.get(alpha_2=language_code).name
        except Exception:
            language_name = "English"

        # Step 3: Retrieve relevant chunks from the document
        # We call retrieve_relevant_chunks directly, NOT the all-in-one query() method
        logger.info(f"Step 3: Retrieving sources for query: '{standalone_query}'")
        context_sources = await self.rag_pipeline.retrieve_relevant_chunks(
            query=standalone_query,
            # document_id_filters=request.document_ids
        )

        # Step 4: Translate the context chunks if necessary
        if context_sources:
            try:
                # This needs the import: from deep_translator import GoogleTranslator
                from deep_translator import GoogleTranslator
                logger.info(f"Step 4: Translating {len(context_sources)} context chunks to '{language_code}'...")
                translator = GoogleTranslator(source='auto', target=language_code)
                for chunk in context_sources:
                    if chunk.get('text'):
                        chunk['text'] = translator.translate(chunk['text'])
                logger.info("Translation of context successful.")
            except Exception as e:
                logger.error(f"Failed to translate context chunks, proceeding with original text. Error: {e}")

        # Step 5: Generate the final answer using the (now translated) context
        final_prompt_query = f"Answer in {language_name}:\n{standalone_query}"
        logger.info(f"Step 5: Generating final answer with prompt: '{final_prompt_query}'")
        final_answer = await self.rag_pipeline.generate_answer(
            query=final_prompt_query,
            context_chunks=context_sources, # Pass the full chunk dictionaries
            llm_provider=request.llm_provider,
            chat_history=chat_history_list,
        )

        
        # This is the code you should add in its place

        await question_store.save_query(user_id, request.query)

        # --- NEW CODE BLOCK STARTS HERE ---
        # Select only the single most relevant source to display to the user.
        top_source = []
        if context_sources:
            # The first item in the list is the most relevant one.
            best_source = context_sources[0]
            
            # Create a clean dictionary with only the fields we want to show.
            top_source.append({
                "document_id": best_source.get("document_id"),
                "page": best_source.get("page"),
                "label": best_source.get("label"),
                "text": best_source.get("text") # Include the relevant text snippet
            })

        return ChatResponse(
            answer=final_answer,
            sources=top_source, # Pass the list containing only the single best source
        )
        

# Example usage (for testing - requires running within an async context if using await)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # This part is tricky to run directly due to async and dependencies
    # You would typically test this through API endpoints or dedicated test scripts
    async def main_test():
        agent = AgentService()
        
        logger.info("--- Testing Chat Query --- ")
        # Test chat query (assuming some documents are already processed and indexed)
        # This requires the vector store to be populated for meaningful RAG results
        chat_req_gemini = ChatRequest(
            query="What is retrieval augmented generation?",
            chat_history=[
                Message(role="user", content="Tell me about RAG."),
                Message(role="assistant", content="Retrieval Augmented Generation combines retrieval with generation models.")
            ],
            llm_provider="gemini" # Test with gemini
        )
        response_gemini = await agent.handle_chat_query(chat_req_gemini)
        logger.info(f"Chat response (Gemini): {response_gemini.answer}")
        if response_gemini.sources:
            logger.info(f"Sources: {response_gemini.sources}")

        chat_req_groq = ChatRequest(
            query="Explain it like I am five.",
            chat_history=[
                Message(role="user", content="Tell me about RAG."),
                Message(role="assistant", content="Retrieval Augmented Generation combines retrieval with generation models."),
                Message(role="user", content="What is retrieval augmented generation?"),
                Message(role="assistant", content=response_gemini.answer) # Continue conversation
            ],
            llm_provider="groq" # Test with groq
        )
        response_groq = await agent.handle_chat_query(chat_req_groq)
        logger.info(f"Chat response (Groq): {response_groq.answer}")
        if response_groq.sources:
            logger.info(f"Sources: {response_groq.sources}")

        # Test document upload (requires a dummy PDF)
        # logger.info("--- Testing Document Upload --- ")
        # dummy_pdf_filename = "test_agent_upload.pdf"
        # dummy_pdf_path = os.path.join(settings.UPLOAD_DIR, dummy_pdf_filename)
        # Ensure UPLOAD_DIR exists: 
        # if not os.path.exists(settings.UPLOAD_DIR):
        #    os.makedirs(settings.UPLOAD_DIR)
        #    logger.info(f"Created UPLOAD_DIR: {settings.UPLOAD_DIR}")

        # if not os.path.exists(dummy_pdf_path):
        #     try:
        #         with open(dummy_pdf_path, "w") as f: # Create a tiny dummy file (not a real PDF)
        #             f.write("dummy pdf content for agent test") 
        #         logger.info(f"Created dummy file for upload test: {dummy_pdf_path}")
        #     except Exception as e:
        #         logger.error(f"Could not create dummy file {dummy_pdf_path}: {e}")
        
        # if os.path.exists(dummy_pdf_path):
        #     upload_response = await agent.handle_document_upload(dummy_pdf_path, dummy_pdf_filename)
        #     logger.info(f"Upload response: {upload_response.message}")
        # else:
        #     logger.warning(f"Skipping upload test, dummy file not found or creatable: {dummy_pdf_path}")

    import asyncio
    try:
        asyncio.run(main_test())
    except Exception as e:
        logger.error(f"Error running main_test: {e}", exc_info=True)
    logger.info("AgentService main_test finished. Run full tests via API or dedicated test scripts.")
