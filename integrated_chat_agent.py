import os
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import openai
import json
import requests
import replicate
import sqlite3
import random

from config import OPENAI_API_KEY
from config import REPLICATE_API_TOKEN
from config import WEATHER_API_KEY

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

class IntegratedChatAgent:
    def __init__(self):
        """Initialize the chat agent without any documents"""
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedding_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize empty vector store and qa chain
        self.vectorstore = None
        self.qa_chain = None
        self._setup_qa_chain()

        print("🤖 Chat Agent with RAG initialized.")
        print("📋 Ready to upload documents using the upload_document() method")

        # Initialize weather functions
        self.weather_functions = [{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    }
                },
                "required": ["location"]
            }
        }]

        # Initialize event and employee database
        self.setup_event_database() 
        self.setup_employee_database() 

        
    def setup_employee_database(self): 
        conn = sqlite3.connect('company.db') 
        c = conn.cursor() 
         
        # Create sample tables 
        c.execute(''' 
            CREATE TABLE IF NOT EXISTS employees ( 
                id INTEGER PRIMARY KEY, 
                name TEXT, 
                department TEXT, 
                salary REAL 
            ) 
        ''') 
         
        c.execute(''' 
            CREATE TABLE IF NOT EXISTS departments ( 
                id INTEGER PRIMARY KEY, 
                name TEXT, 
                budget REAL 
            ) 
        ''') 
         
        # Insert sample data 
        c.execute("INSERT OR IGNORE INTO employees VALUES (1, 'John Doe', 'Engineering', 75000)") 
        c.execute("INSERT OR IGNORE INTO employees VALUES (2, 'Jane Smith', 'Marketing', 65000)") 
        c.execute("INSERT OR IGNORE INTO departments VALUES (1, 'Engineering', 1000000)") 
        c.execute("INSERT OR IGNORE INTO departments VALUES (2, 'Marketing', 500000)") 
         
        conn.commit() 
        conn.close() 

    def setup_event_database(self): 
        conn = sqlite3.connect('events.db') 
        c = conn.cursor() 
         
        c.execute(''' 
            CREATE TABLE IF NOT EXISTS events ( 
                id INTEGER PRIMARY KEY, 
                name TEXT, 
                type TEXT,  -- 'indoor' or 'outdoor' 
                description TEXT, 
                location TEXT, 
                date TEXT 
            ) 
        ''') 
         
        # Sample events 
        events = [ 
            ('Summer Concert', 'outdoor', 'Live music in the park', 'Central Park', '2025-07-15'), 
            ('Art Exhibition', 'indoor', 'Modern art showcase', 'City Gallery', '2025-07-15'), 
            ('Food Festival', 'outdoor', 'International cuisine', 'Waterfront', '2025-07-16'), 
            ('Theater Show', 'indoor', 'Classical drama', 'Grand Theater', '2025-07-16') 
        ] 
         
        c.executemany('INSERT OR IGNORE INTO events (name, type, description, location, date) VALUES (?,?,?,?,?)', events) 
        conn.commit() 
        conn.close() 

    def _generate_sql(self, question: str) -> str:
        """Generate SQL from natural language"""
        schema = """
        Table: employees
        Columns:
        - id (INTEGER PRIMARY KEY)
        - name (TEXT)
        - department (TEXT)
        - salary (REAL)

        Table: departments
        Columns:
        - id (INTEGER PRIMARY KEY)
        - name (TEXT)
        - budget (REAL)
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""You are a SQL expert. Use this schema:\n{schema}
                Return ONLY the SQL query without any explanation or markdown formatting."""},
                {"role": "user", "content": f"Generate SQL for: {question}"}
            ]
        )
        
        sql = response.choices[0].message.content.strip()
        sql = sql.replace('```sql', '').replace('```SQL', '').replace('```', '').strip()
        
        return sql
    
    def _execute_query(self, sql: str):
        """Execute SQL query safely"""
        # Safety check
        sql_lower = sql.lower()
        if any(word in sql_lower for word in ['drop', 'delete', 'update', 'insert']):
            return "❌ Only SELECT queries are allowed for security reasons."
        
        conn = sqlite3.connect('company.db')
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"❌ SQL execution error: {str(e)}"
        finally:
            conn.close()

    def _format_results(self, results) -> str:
        """Format query results for display"""
        if not isinstance(results, list):
            return str(results)
        if not results:
            return "No results found"
        
        if len(results[0]) == 1:
            return "\n".join([str(row[0]) for row in results])
        
        formatted_rows = []
        for row in results:
            row_items = []
            for item in row:
                if isinstance(item, float):
                    if any(keyword in str(row).lower() for keyword in ['salary', 'budget']):
                        row_items.append(f"${item:,.2f}")
                    else:
                        row_items.append(f"{item:.2f}")
                else:
                    row_items.append(str(item))
            formatted_rows.append(" | ".join(row_items))
        
        return "\n".join(formatted_rows)

    def route_query(self, user_input: str) -> str:
        """
        Intelligent routing system to determine which agent to use
        """
        routing_prompt = f"""
        Analyze the following user query and determine which type of response is most appropriate.
        
        User query: "{user_input}"
        
        Choose ONE of the following categories:
        1. "general" - General conversation, greetings, questions about uploaded documents/reports, or other topics
        2. "weather" - Weather-related queries or questions about current conditions
        3. "database" - SQL queries only about employees and their salary (not about external documents or other company related information eg. revenue, profit & lost, financial, etc)
        4. "image" - Requests to generate, create, or make images/pictures
        5. "recommendation" - Event recommendations, activity suggestions based on weather/date
        
        Respond with only the category name (one word).
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": routing_prompt}],
                temperature=1
            )
            route = response.choices[0].message.content.strip().lower()
            
            # Validate route
            valid_routes = ["weather", "database", "image", "recommendation", "general"]
            if route not in valid_routes:
                route = "general"
                
            return route
        except Exception as e:
            print(f"❌ Routing error: {str(e)}")
            return "general"   

    def _get_weather(self, location: str, date: str = None) -> dict:
        """Get weather data from API for specific date"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if date is None or date == today:
            # Current weather for today
            url = "http://api.weatherapi.com/v1/current.json"
            params = {
                "key": WEATHER_API_KEY,
                "q": location,
                "aqi": "no"
            }
        else:
            # Forecast or historical weather for specific date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            days_diff = (target_date - datetime.now()).days
            
            if days_diff <= 10 and days_diff >= 0:
                # Future forecast (up to 10 days)
                url = "http://api.weatherapi.com/v1/forecast.json"
                params = {
                    "key": WEATHER_API_KEY,
                    "q": location,
                    "days": max(1, days_diff + 1),
                    "aqi": "no",
                    "alerts": "no"
                }
            else:
                # For dates beyond 10 days or past dates, use current weather as fallback
                print(f"⚠️ Weather forecast not available for {date}, using current weather")
                url = "http://api.weatherapi.com/v1/current.json"
                params = {
                    "key": WEATHER_API_KEY,
                    "q": location,
                    "aqi": "no"
                }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        # Extract relevant weather for the specific date
        if 'forecast' in weather_data and date != today:
            # Find the forecast for the specific date
            for forecast_day in weather_data['forecast']['forecastday']:
                if forecast_day['date'] == date:
                    # Create a structure similar to current weather
                    return {
                        'location': weather_data['location'],
                        'current': {
                            'condition': forecast_day['day']['condition'],
                            'temp_c': forecast_day['day']['avgtemp_c'],
                            'temp_f': forecast_day['day']['avgtemp_f'],
                            'humidity': forecast_day['day']['avghumidity'],
                            'wind_kph': forecast_day['day']['maxwind_kph']
                        },
                        'forecast_date': date
                    }
        
        return weather_data
    
    def _get_events(self, date, event_type=None): 
        conn = sqlite3.connect('events.db') 
        c = conn.cursor() 
         
        try: 
            if event_type: 
                c.execute('SELECT * FROM events WHERE date = ? AND type = ?', (date, event_type)) 
            else: 
                c.execute('SELECT * FROM events WHERE date = ?', (date,)) 
                 
            events = c.fetchall() 
            return events 
        except sqlite3.Error as e: 
            raise Exception(f"Database error: {str(e)}") 
        finally: 
            conn.close() 

    def _generate_recommendations(self, weather_data: dict, events: List[Tuple], location: str, date: str) -> str:
        """Generate event recommendations based on weather and available events"""
        try:
            weather_condition = weather_data['current']['condition']['text']
            temperature = weather_data['current']['temp_c']
            
            context = f"""
            Location: {location}
            Date: {date}
            Weather: {weather_condition}
            Temperature: {temperature}°C
            
            Available events:
            """
            
            for event in events:
                context += f"- {event[1]} ({event[2]}): {event[3]} at {event[4]}\n"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a helpful event recommender. Consider the weather conditions 
                    and suggest suitable events. For outdoor events, consider the temperature and weather conditions. 
                    Be specific about why you recommend certain events over others. Include emojis and make it engaging.
                    Format your response with clear recommendations and reasoning."""},
                    {"role": "user", "content": context}
                ]
            )
            
            recommendation = response.choices[0].message.content
            
            return f"🎯 **Event Recommendations for {location} on {date}**\n\n{recommendation}"
            
        except Exception as e:
            return f"❌ Recommendation generation error: {str(e)}"

    def handle_recommendation_query(self, query: str) -> str:
        """Handle event recommendation requests"""
        try:
            # Extract location and date from query
            location = "Singapore"  # Default location
            date = datetime.now().strftime("%Y-%m-%d")  # Default to today
            
            # Try to extract location from query
            location_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"Extract the location mentioned in this query. If no location is mentioned, return 'Singapore'. Query: {query}"
                }]
            )
            
            extracted_location = location_response.choices[0].message.content.strip()
            if extracted_location and extracted_location.lower() != "singapore":
                location = extracted_location
            print(f"🌍 Location: {location}")
            
            # Try to extract date from query
            date_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"""Extract the date mentioned in this query and convert it to YYYY-MM-DD format. 
                    Consider today's date is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A')}).
                    
                    Examples:
                    - "today" -> {datetime.now().strftime('%Y-%m-%d')}
                    - "tomorrow" -> {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                    - "July 15" -> 2025-07-15
                    - "next Monday" -> calculate the next Monday's date
                    
                    If no date is mentioned, return '{datetime.now().strftime('%Y-%m-%d')}' (today).
                    
                    Query: {query}
                    
                    Respond with only the date in YYYY-MM-DD format."""
                }]
            )
            
            extracted_date = date_response.choices[0].message.content.strip()
            # Validate date format
            try:
                datetime.strptime(extracted_date, '%Y-%m-%d')
                date = extracted_date
                print(f"📅 Extracted date: {date}")
            except ValueError:
                # If parsing fails, keep today's date as default
                print(f"📅 Using default date (today): {date}")
            
            # Get weather and events for the specific date
            weather_data = self._get_weather(location, date)
            events = self._get_events(date)
            
            if not events:
                return f"❌ No events found for {date}. Please try a different date."
            
            # Generate recommendations
            recommendations = self._generate_recommendations(weather_data, events, location, date)
            
            return recommendations
            
        except Exception as e:
            return f"❌ Recommendation error: {str(e)}"


    def handle_database_query(self, query: str) -> str:
        """Handle database/SQL queries"""
        try:
            # Generate SQL from natural language
            sql = self._generate_sql(query)
            
            if not sql:
                return "❌ I couldn't generate a valid SQL query from your request."
            
            # Execute query safely
            results = self._execute_query(sql)
            
            if isinstance(results, str):  # Error case
                return results
                
            # Format results
            formatted_results = self._format_results(results)
            
            return f"📊 Query: {query}\n\n🔍 SQL: {sql}\n\n📋 Results:\n{formatted_results}"
            
        except Exception as e:
            return f"❌ Database query error: {str(e)}"


    def handle_weather_query(self, query: str) -> str:
        """Handle weather-related queries"""
        try:
            # Extract date from weather query
            date = datetime.now().strftime("%Y-%m-%d")  # Default to today
            
            date_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"""Extract the date mentioned in this weather query and convert it to YYYY-MM-DD format. 
                    Consider today's date is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A')}).
                    
                    Examples:
                    - "today" -> {datetime.now().strftime('%Y-%m-%d')}
                    - "tomorrow" -> {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                    
                    If no date is mentioned, return '{datetime.now().strftime('%Y-%m-%d')}' (today).
                    
                    Query: {query}
                    
                    Respond with only the date in YYYY-MM-DD format."""
                }]
            )
            
            extracted_date = date_response.choices[0].message.content.strip()
            try:
                datetime.strptime(extracted_date, '%Y-%m-%d')
                date = extracted_date
            except ValueError:
                pass  # Keep today's date as default
            
            # Use OpenAI to extract location and get weather
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": query}],
                tools=[{"type": "function", "function": func} for func in self.weather_functions],
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "get_weather":
                    arguments = json.loads(tool_call.function.arguments)
                    weather_data = self._get_weather(arguments["location"], date)
                    
                    date_text = "today" if date == datetime.now().strftime("%Y-%m-%d") else f"on {date}"
                    forecast_indicator = "" if 'forecast_date' not in weather_data else "🔮 Forecast: "
                    
                    return (f"🌤️ {forecast_indicator}Weather in {weather_data['location']['name']} {date_text}: "
                           f"{weather_data['current']['temp_c']}°C, "
                           f"{weather_data['current']['condition']['text']}. "
                           f"Humidity: {weather_data['current']['humidity']}%")
            else:
                return "❌ I couldn't determine the location from your weather query. Please specify a city or location."
                
        except Exception as e:
            return f"❌ Weather query error: {str(e)}"

    def handle_image_query(self, query: str) -> str:
        """Handle text-to-image generation requests"""
        try:
            
            # Generate image using Replicate
            output = replicate.run(
               "google/imagen-4",
                 input={
                     "prompt": query,
                     "seed": random.randint(1, 1000000),  
                     "steps": 30,
                     "format": "png"
                    }
                 )
                        
            if output:
                return f"🖼️ Image generated successfully!\n\n📝 Image URL: {output}\n\n💡 Tip: Click the URL to view your generated image!"
            else:
                return "❌ Failed to generate image. Please try again with a different prompt."
                
        except Exception as e:
            return f"❌ Image generation error: {str(e)}"
    
    
    def load_document(self, document_path):
        """Load a single document and create/update the vector store"""
        try:
            # Load the document
            loader = PyMuPDFLoader(document_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if self.vectorstore is None:
                # Create new vector store
                self.vectorstore = Chroma.from_documents(chunks, self.embedding_model)
            else:
                # Add to existing vector store
                self.vectorstore.add_documents(chunks)
            
            # Update QA chain
            self._setup_qa_chain()
            
            print(f"✅ Successfully loaded document: {document_path}")
            print(f"📄 Added {len(chunks)} chunks to the knowledge base")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading document {document_path}: {str(e)}")
            return False
    
    def upload_document(self, document_path):
        """Public method to upload a new document"""
        if not os.path.exists(document_path):
            print(f"❌ File not found: {document_path}")
            return False
        
        return self.load_document(document_path)
    
    def _setup_qa_chain(self):
        """Set up the conversational retrieval chain"""
        if self.vectorstore is not None:
            # With document retrieval
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                self.vectorstore.as_retriever(),
                memory=self.memory
            )
        else:
            # Create empty vectorstore for consistent interface
            empty_vectorstore = Chroma.from_texts(
                texts=["I am a helpful assistant ready to answer questions."], 
                embedding=self.embedding_model
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                empty_vectorstore.as_retriever(),
                memory=self.memory
            )

    def handle_general_query(self, question):
        """Ask a question to the chat agent"""
        if self.qa_chain is None:
            return "chat agent is not initialized"
        
        try:
            # Always use the same interface - ConversationalRetrievalChain
            result = self.qa_chain.invoke({"question": question, "chat_history": []})
            return result["answer"]
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"

    def chat_agent(self, question):
       """
        Main chat interface that routes queries to appropriate handlers
        """
       if not question.strip():
            return "Please enter a message."
        
       try:
            # Route the query
            route = self.route_query(question)
            print(f"🔄 Routing to: {route}")
            
            # Handle based on route
            if route == "weather":
                response = self.handle_weather_query(question)
            elif route == "image":
                response = self.handle_image_query(question)
            elif route == "database":
                response = self.handle_database_query(question)
            elif route == "recommendation":
                response = self.handle_recommendation_query(question)
            else:  # general
                response = self.handle_general_query(question)
            
            # Save to memory for general conversations and document queries
            if route in ["general"]:
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
            
       except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    

def main():
    """Main function to run the enhanced chat agent"""
    agent = IntegratedChatAgent()
    print("\n=== TESTING: ask question without uploading document ===")

    demo_questions = [
            "What is the capital of France?",
            "What is the weather like in Tokyo?",
            "What's the weather tomorrow in New York?",
            "Generate an image of a cat",
            "Show me all employees earning more than 70000",
            "What are the key BP's financial highlights in the annual report?",
            "Recommend events for today in Singapore",
            "What events are available tomorrow?",
            "Suggest activities for July 15th in Tokyo"
        ]
    for question in demo_questions:
            print(f"\n👤 User: {question}")
            answer = agent.chat_agent(question)
            print(f"🤖 Assistant: {answer}")
    

    # Test functionality by uploading the test document
    print("\n=== TESTING: Upload Functionality ===")
    test_document = "./bp-annual-report-and-form-20f-2024.pdf"
    
    if os.path.exists(test_document):
        print(f"📤 Testing upload with: {test_document}")
        agent.upload_document(test_document)
        
        # Demo the original functionality after upload
        print("\n=== TESTING: ask question after uploading document ===")
        demo_questions = [
            "What are the key BP's financial highlights in the annual report?",
            "Can you tell me more about the BP's annual report revenue?",
            "What about the BP's annual report profit?"
        ]
        
        for question in demo_questions:
            print(f"\n👤 User: {question}")
            answer = agent.chat_agent(question)
            print(f"🤖 Assistant: {answer}")
    else:
        print(f"⚠️  Test document not found: {test_document}")

    
if __name__ == "__main__":
    main()
