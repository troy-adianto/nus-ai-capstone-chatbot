import streamlit as st
import os
import tempfile
from integrated_chat_agent import IntegratedChatAgent

# Set page config
st.set_page_config(
    page_title="AI Chat Agent with RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = IntegratedChatAgent()
    st.session_state.chat_history = []
    st.session_state.uploaded_docs = []

# Sidebar for document upload
with st.sidebar:
    st.header("📂 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        help="Upload PDF files to enhance the AI's knowledge base"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Upload to agent
        if st.button("📤 Upload Document"):
            with st.spinner("Processing document..."):
                success = st.session_state.agent.upload_document(temp_path)
                
                if success:
                    st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                    st.session_state.uploaded_docs.append(uploaded_file.name)
                else:
                    st.error("❌ Failed to upload document")
                
                # Clean up temp file
                os.unlink(temp_path)
    
    # Show uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("📋 Uploaded Documents")
        for doc in st.session_state.uploaded_docs:
            st.write(f"• {doc}")

    st.divider()
    
    # Agent capabilities
    st.header("🔧 Agent Capabilities")
    st.write("""
    **Available Features:**
    - 💬 General conversation
    - 📄 Document Q&A (RAG)
    - 🌤️ Weather information
    - 🗄️ Database queries
    - 🎨 Image generation
    - 🎯 Event recommendations
    """)

# Main chat interface
st.title("🤖 AI Chat Agent with RAG")
st.write("Ask questions, get weather info, query databases, generate images, or get event recommendations!")

# Quick upload for BP Annual Report
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚡ Quick Start with Sample Document")
    st.write("Load the BP Annual Report 2024 to test document Q&A capabilities:")

with col2:
    bp_report_path = "./bp-annual-report-and-form-20f-2024.pdf"
    bp_report_name = "BP Annual Report 2024"
    
    if os.path.exists(bp_report_path):
        if st.button("📊 Load BP Annual Report", 
                    help="Upload the BP Annual Report & Form 20-F 2024", 
                    type="primary",
                    use_container_width=True):
            with st.spinner("Loading BP Annual Report..."):
                success = st.session_state.agent.upload_document(bp_report_path)
                
                if success:
                    st.success(f"✅ Successfully loaded: {bp_report_name}")
                    if bp_report_name not in st.session_state.uploaded_docs:
                        st.session_state.uploaded_docs.append(bp_report_name)
                    st.rerun()  # Refresh to show the updated document list
                else:
                    st.error("❌ Failed to load BP Annual Report")
    else:
        st.info("💡 BP Annual Report not found in project directory")

# Show loaded documents status in main area
if st.session_state.uploaded_docs:
    st.info(f"📄 **Loaded Documents:** {', '.join(st.session_state.uploaded_docs)}")

st.divider()

# Display chat history
chat_container = st.container()

with chat_container:
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        # User message
        st.chat_message("user").write(question)
        
        # Agent response
        st.chat_message("assistant").write(answer)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.chat_history.append((prompt, ""))
    
    # Display user message immediately
    st.chat_message("user").write(prompt)
    
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.chat_agent(prompt)
                st.write(response)
                
                # Update chat history with response
                st.session_state.chat_history[-1] = (prompt, response)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history[-1] = (prompt, error_msg)

# Example questions
st.divider()
st.subheader("💡 Example Questions")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**📄 Document Questions:**")
    st.caption("💡 First load BP Annual Report using the button above!")
    
    if st.button("Key financial highlights"):
        st.session_state.example_question = "What are the key BP's financial highlights in the annual report?"
    
    if st.button("Revenue performance"):
        st.session_state.example_question = "Can you tell me more about the BP's annual report revenue?"
    
    if st.button("Risk factors"):
        st.session_state.example_question = "What are the main risk factors mentioned in BP's annual report?"

with col2:
    st.write("**🌤️ Weather & Events:**")
    if st.button("Weather in Tokyo"):
        st.session_state.example_question = "What's the weather like in Tokyo?"
    
    if st.button("Events in Tokyo"):
        st.session_state.example_question = "Suggest activities for July 15th in Tokyo"

with col3:
    st.write("**🗄️ Database & Images:**")
    if st.button("Show employees earning more than 70000"):
        st.session_state.example_question = "Show me all employees earning more than 70000"
    
    if st.button("Generate image of Marina Bay Sands over the clouds"):
        st.session_state.example_question = "Generate an image of a Marina Bay Sands over the clouds"

# Handle example questions
if 'example_question' in st.session_state:
    question = st.session_state.example_question
    del st.session_state.example_question
    
    # Add to chat history
    st.session_state.chat_history.append((question, ""))
    
    # Get response
    try:
        response = st.session_state.agent.chat_agent(question)
        st.session_state.chat_history[-1] = (question, response)
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.chat_history[-1] = (question, error_msg)
    
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🤖 AI Chat Agent with RAG - Powered by OpenAI & LangChain
</div>
""", unsafe_allow_html=True)
