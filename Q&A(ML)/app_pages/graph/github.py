import re
import requests
import base64
from markdown import markdown
from bs4 import BeautifulSoup
from groq import Groq
from langchain_core.tools import tool

# Initialize Groq client
client1 = Groq(api_key="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")

# List of key directories developers often look for
KEY_DIRECTORIES = ['src', 'tests', 'docs', 'examples']

# Function to convert markdown to plain text
def md_to_text(md: str) -> str:
    """Convert markdown text to plain text for summarization."""
    try:
        html = markdown(md)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Failed to convert markdown to text: {e}")
        return md  # Return original text as fallback

# Function to summarize text with a developer-focused prompt
def summarize_text(text: str) -> str:
    """Summarize the README, focusing on purpose, installation, and usage."""
    try:
        response = client1.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Summarize the following README, focusing on the purpose, installation instructions, and usage examples:\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to summarize text: {e}")
        return "Failed to generate summary."

# Main tool function to summarize GitHub repository
@tool(parse_docstring=True)
def github_summarize_repo(url: str, keyword: str = "") -> tuple:  # Changed default to empty string
    """
    Summarizes a GitHub repository with a developer-friendly overview, including README summary,
    key directory structure, and programming languages. Optionally searches for a keyword in the README.

    Args:
        url (str): The URL of the GitHub repository (e.g., https://github.com/username/repo_name).
        keyword (str): A keyword to search for in the README (optional, default: empty string).

    Returns:
        tuple: A tuple containing the results string and a dictionary with intermediate outputs.
              Format: (results, {"intermediate_outputs": [{"output": results}]})
    """
    try:
        # Extract username and repo_name from URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if not match:
            return "Invalid GitHub repository URL. Please provide a URL in the format 'https://github.com/username/repo_name'.", {
                "intermediate_outputs": [{"output": "Invalid URL"}]
            }
        username, repo_name = match.groups()

        # Fetch repository details (default branch and description)
        repo_url = f"https://api.github.com/repos/{username}/{repo_name}"
        response = requests.get(repo_url, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code != 200:
            return f"Failed to fetch repository details: HTTP {response.status_code}. The repository might be private or not exist.", {
                "intermediate_outputs": [{"output": f"Failed to fetch repo details: {response.status_code}"}]
            }
        repo_data = response.json()
        default_branch = repo_data['default_branch']
        description = repo_data.get('description', 'No description provided.')

        # Fetch README
        readme_url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
        response = requests.get(readme_url, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code == 404:
            readme_text = "No README found in the repository."
        elif response.status_code != 200:
            return f"Failed to fetch README: HTTP {response.status_code}.", {
                "intermediate_outputs": [{"output": f"Failed to fetch README: {response.status_code}"}]
            }
        else:
            readme_data = response.json()
            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
            readme_text = md_to_text(readme_content)

        # Summarize README if available
        if readme_text != "No README found in the repository.":
            summary = summarize_text(readme_text)
        else:
            summary = "No README available to summarize."

        # Fetch top-level file tree using the default branch
        tree_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/{default_branch}"
        response = requests.get(tree_url, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code == 404:
            structure = "Repository tree not found. The repository might be empty."
        elif response.status_code != 200:
            structure = f"Failed to fetch repository structure: HTTP {response.status_code}."
        else:
            tree_data = response.json()
            files = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
            dirs = [item['path'] for item in tree_data['tree'] if item['type'] == 'tree']
            structure = f"Top-level files: {', '.join(files)}\nTop-level directories: {', '.join(dirs)}"
            key_dirs_present = [d for d in dirs if d in KEY_DIRECTORIES]
            if key_dirs_present:
                structure += f"\nKey directories present: {', '.join(key_dirs_present)}"
            else:
                structure += "\nNo key directories (src, tests, docs, examples) found."

        # Fetch programming languages
        languages_url = f"https://api.github.com/repos/{username}/{repo_name}/languages"
        response = requests.get(languages_url, headers={"Accept": "application/vnd.github.v3+json"})
        if response.status_code == 200:
            languages_data = response.json()
            languages = ', '.join(languages_data.keys()) or "No languages detected."
        else:
            languages = "Failed to fetch programming languages."

        # Compile results in a developer-friendly format
        results = (
            f"Description: {description}\n\n"
            f"Summary of README:\n{summary}\n\n"
            f"Repository Structure:\n{structure}\n\n"
            f"Programming Languages: {languages}"
        )

        # Search for keyword if provided and README exists
        if keyword.strip() and readme_text != "No README found in the repository.":  # Check for non-empty keyword
            if keyword.lower() in readme_text.lower():
                results += f"\nKeyword '{keyword}' found in README."
            else:
                results += f"\nKeyword '{keyword}' not found in README."

        return results, {"intermediate_outputs": [{"output": results}]}

    except Exception as e:
        return f"Error processing repository: {str(e)}", {
            "intermediate_outputs": [{"output": str(e)}]
        }