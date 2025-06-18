markdown
# Matchmaker AI - Resume Screening Tool

![Matchmaker AI Logo](static/logo.jpeg)

Matchmaker AI is an intelligent resume screening application that helps recruiters and hiring managers quickly evaluate candidate resumes against job descriptions using natural language processing and machine learning techniques.

## Features

- **BERT-based semantic matching** - Understands context beyond simple keyword matching
- **Multi-document processing** - Analyze multiple resumes simultaneously
- **Comprehensive scoring** - Breaks down match scores into base similarity, keyword overlap, and exact matches
- **Visual analytics** - Interactive charts showing skill distribution and score ranges
- **Export capabilities** - Download all results to Excel for further analysis
- **Customizable weights** - Adjust the importance of different matching factors

## Technology Stack

- **Backend**: Python with Flask
- **NLP Libraries**: spaCy, BERT (Hugging Face Transformers)
- **Document Processing**: PyPDF2, pdfplumber, python-docx
- **Frontend**: Bootstrap 5, Chart.js
- **Data Export**: SheetJS (xlsx)

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sdarmaraj83/matchmakerAI.git
cd matchmakerAI
Step 2: Create a Virtual Environment (Recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Step 3: Install Required Packages
bash
pip install -r requirements.txt
Step 4: Download NLP Models
The application requires several NLP models:

spaCy English Model (for keyword extraction):

bash
python -m spacy download en_core_web_sm
BERT Model (for semantic matching - will be automatically downloaded on first run):

The application uses bert-base-uncased from Hugging Face

Approximately 440MB will be downloaded to your cache directory (~/.cache/huggingface)

Step 5: Set Up Required Folders
The application expects the following folder structure:

text
matchmakerAI/
├── app_rc_M.py          # Main application file
├── requirements.txt     # Python dependencies
├── static/              # Static files (CSS, JS, images)
│   └── logo.jpeg        # Application logo
├── templates/           # HTML templates
│   └── index.html       # Main interface
├── logs/                # Application logs
├── uploads/             # Temporary file uploads (created automatically)
└── README.md            # This documentation
Create the necessary folders if they don't exist:

bash
mkdir -p static templates logs
Running the Application
bash
python app_rc_M.py
The application will start on http://localhost:5000. Open this URL in your web browser to access the interface.

Usage
Enter a job title and description in the left panel

Upload one or more resume files (PDF or DOCX)

Adjust the matching parameters if needed:

Similarity Threshold

Keyword Boost Weight

Exact Match Weight

Click "Process Resumes" to analyze the documents

View results including:

Match scores for each resume

Top keywords from the job description

Skills distribution across candidates

Score distribution histogram

Export all data to Excel using the download button

Contributing
We welcome contributions to Matchmaker AI! Here's how you can help:

Reporting Issues
If you find any bugs or have suggestions for improvements, please open an issue on GitHub.

Making Changes
Fork the repository

Create a new branch for your feature/fix:

bash
git checkout -b feature/your-feature-name
Make your changes and commit them:

bash
git commit -m "Description of your changes"
Push to your fork:

bash
git push origin feature/your-feature-name
Create a pull request on the main repository

Pull Request Guidelines
Ensure your code follows PEP 8 style guidelines

Include tests for new features if applicable

Update documentation as needed

Keep commits focused and logical

Known Limitations
Large PDF files (>10MB) may take longer to process

Image-based PDFs without text layers may not be processed correctly

Very long documents may be truncated due to BERT's 512-token limit

License
Matchmaker AI is released under the MIT License. See LICENSE file for details.

Contact
For questions or support, please contact the maintainer at [your email].

text

Key formatting notes for GitHub's markdown:
1. Used `#`, `##`, `###` for headers
2. Used triple backticks for code blocks
3. Used `-` for unordered lists
4. Added proper links to GitHub issues and PRs
5. Maintained the folder structure visualization with code blocks
6. Included relative paths for local files (static/logo.jpeg)
7. Added placeholder for LICENSE file link

