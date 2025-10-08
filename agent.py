import pandas as pd
from ddgs import DDGS
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import sys
import os
from typing import Dict, Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm

def web_search_function(query: str, max_results: int = 50) -> str:
    """
    Performs a DuckDuckGo search and saves results to a CSV file.
    
    Args:
        context: domain_classification from the router agent
        max_results: Maximum number of search results to fetch
    
    Returns:
        String with the CSV filename or error message
    """
    try:
        print(f"\n[WEB_SEARCH_FUNCTION] Searching for: {query}")
        
        
        with DDGS() as ddgs:
            results = ddgs.text(
                query=query,
                region='wt-wt',
                safesearch='off',
                timelimit='w',
                max_results=max_results
            )

            if not results:
                return "ERROR: No results found"

            results_df = pd.DataFrame(results)
            
            # Create domain-specific filename
            output_filename = f'search_results_{query.lower().replace("-", "_")}.csv'
            results_df.to_csv(output_filename, index=False)
            
            print(f"[WEB_SEARCH_FUNCTION] ✓ Saved {len(results)} results to '{output_filename}'")
            
            return output_filename

    except Exception as e:
        print(f"[WEB_SEARCH_FUNCTION] ✗ Error: {e}", file=sys.stderr)
        return f"ERROR: {str(e)}"


def web_scrape_function(input_csv: str, domain: str) -> str:
    """
    Scrapes websites from URLs in a CSV file using Playwright.
    
    Args:
        input_csv: Path to CSV file containing URLs
        query: web_search results CSV filename
    
    Returns:
        String with the scraped content CSV filename or error message
    """
    if not os.path.exists(input_csv):
        return f"ERROR: Input file '{input_csv}' not found"

    try:
        print(f"\n[WEB_SCRAPE_FUNCTION] Reading URLs from '{input_csv}'...")
        df = pd.read_csv(input_csv)
        
        if 'href' not in df.columns:
            return "ERROR: CSV must contain 'href' column"

        scraped_data = []
        error_count = 0
        
        print(f"[WEB_SCRAPE_FUNCTION] Starting to scrape {len(df)} URLs...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for index, row in df.iterrows():
                url = row['href']
                print(f"[WEB_SCRAPE_FUNCTION] Scraping ({index + 1}/{len(df)}): {url[:60]}...")

                try:
                    page.goto(url, timeout=60000)
                    page.wait_for_load_state('domcontentloaded')
                    body_text = page.locator('body').inner_text()
                    scraped_data.append({'url': url, 'scraped_text': body_text})

                except PlaywrightTimeoutError:
                    print(f"  ⚠ Timeout for: {url[:60]}")
                    scraped_data.append({'url': url, 'scraped_text': 'TIMEOUT_ERROR'})
                    error_count += 1
                except Exception as e:
                    print(f"  ⚠ Error for: {url[:60]}")
                    scraped_data.append({'url': url, 'scraped_text': f'ERROR: {e}'})
                    error_count += 1

            browser.close()

        if scraped_data:
            output_filename = f'scraped_content_{domain.lower().replace("-", "_")}.csv'
            scraped_df = pd.DataFrame(scraped_data)
            scraped_df.to_csv(output_filename, index=False)
            
            success_count = len(scraped_data) - error_count
            print(f"[WEB_SCRAPE_FUNCTION] ✓ Saved scraped content to '{output_filename}'")
            print(f"[WEB_SCRAPE_FUNCTION] ✓ Successfully scraped: {success_count}/{len(df)} URLs")
            
            return f"{output_filename} (Success: {success_count}, Errors: {error_count})"
        else:
            return "ERROR: No data was scraped"

    except Exception as e:
        print(f"[WEB_SCRAPE_FUNCTION] ✗ Error: {e}", file=sys.stderr)
        return f"ERROR: {str(e)}"


OLLAMA_MODEL = LiteLlm(model="ollama_chat/gemma3")

router_agent = LlmAgent(
    name="RouterAgent",
    model=OLLAMA_MODEL,
    instruction="""Receive and Analyze the Query: Take the user's query as input and carefully examine its content to understand the core subject and intent.
Classification Rules: 
Follow these steps:
1.If the query is primarily about artificial intelligence, machine learning, robotics, automation, neural networks, or autonomous systems, classify it as "AI-Robotics".
2.If the query is primarily about international relations, foreign policy, global conflicts, treaties, or the political and economic relationships between countries, classify it as "geopolitics".
3.If the query clearly contains elements of both AI-Robotics and geopolitics (for example, "the role of AI in international espionage" or "the global race for semiconductor supremacy"), classify it as both.
4.If the query does not fit into any of the above categories, classify it as "out-of-domain".
5.Output Formatting: The output you generate must strictly follow the markdown formats specified below. Do not add any other text, greetings, or explanations.
6.For In-Domain Queries (AI-Robotics, Geopolitics, or Both):Your output must contain two lines: "Domain" and "Context".
The "Domain" line should specify the identified domain(s). If the query fits both, list them separated by a comma.
The "Context" line should provide a concise summary of the user's query, capturing its main topic or question.
7.For Out-of-Domain Queries:
Your output must be a two lines specifying that the query is out-of-domain and the query itself
8. **DO NOT GIVE ANSWER BY YOURSELF** Just provide the classification and context.""",
    description="Classifies user queries into AI-Robotics, Geopolitics, both, or out-of-domain.",
    output_key="domain_classification"
)


# --- STEP 2: WEB SEARCH AGENT ---
# Searches the web based on domain and context from router
web_search_agent = LlmAgent(
    name="WebSearchAgent",
    model=OLLAMA_MODEL,
    instruction="""You are an expert web search agent that searches for information based on the classified domain.

**classification to search**
{domain_classification}

Your task:
1. Check the domain value
2. If domain is "out-of-domain", output: "Cannot search: Query is out-of-domain"
3. Otherwise, use the web_search_function tool with:
   - query parameter: use the context value
   - domain parameter: use the first domain from domain (if multiple domains like "AI-Robotics, Geopolitics", use "AI-Robotics")
   - max_results: 50

Search Guidelines by Domain:
- AI-Robotics: Focus on AI, machine learning, robotics, automation terms
- Geopolitics: Focus on international relations, foreign policy, diplomacy terms
- Add time-relevant keywords: recent, latest, 2025, current

After the tool returns a filename, output:
Search completed: [filename]

IMPORTANT: You MUST call the web_search_function tool. Do not just describe what you would do.""",
    description="Searches the web based on classified domain and returns CSV with URLs.",
    tools=[FunctionTool(web_search_function)],
    output_key="search_csv"
)


# --- STEP 3: WEB SCRAPING AGENT ---
# Scrapes content from URLs in the search results CSV
web_scraping_agent = LlmAgent(
    name="WebScrapingAgent",
    model=OLLAMA_MODEL,
    instruction="""You are a web scraping agent that extracts content from URLs.

**Original Classification**
{domain_classification}

**Search Results CSV**
{search_csv}

Your task:
1. Check if search_csv starts with "ERROR" or contains "out-of-domain"
   - If yes, output: "Cannot scrape: No valid search results"
2. Otherwise, use the web_scrape_function tool with:
   - input_csv parameter: use the {search_csv} value
   - domain parameter: use the first domain from domain

The tool will:
- Read all URLs from the CSV file
- Scrape text content from each webpage
- Handle timeouts and errors automatically
- Save results to a new CSV file

After the tool completes, output:
Scraping completed: [result message from tool]

IMPORTANT: You MUST call the web_scrape_function tool. Do not just describe what you would do.""",
    description="Scrapes content from URLs in the search results CSV.",
    tools=[FunctionTool(web_scrape_function)],
    output_key="scrape_csv"
)



# This is the main sequential workflow that runs: Router → Search → Scrape
root_agent = SequentialAgent(
    name="SearchScrapePipeline",
    sub_agents=[
        router_agent,
        web_search_agent,
        web_scraping_agent
    ],
    description="Sequential workflow: Classifies query → Searches web → Scrapes content"
)