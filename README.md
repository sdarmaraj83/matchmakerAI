
# Matchmaker AI - Resume Screening Tool

![Matchmaker AI Logo](https://raw.githubusercontent.com/sdarmaraj83/matchmakerAI/main/static/logo.jpeg)

Matchmaker AI is an intelligent resume screening application that helps recruiters and hiring managers quickly evaluate candidate resumes against job descriptions using natural language processing and machine learning techniques.

![App UI Screenshot](https://raw.githubusercontent.com/sdarmaraj83/matchmakerAI/main/static/ui_screenshot.png)





---

## Features

- **BERT-based semantic matching** - Understands context beyond simple keyword matching
- **Multi-document processing** - Analyze multiple resumes simultaneously
- **Comprehensive scoring** - Match score breakdown: base similarity, keyword overlap, and exact matches
- **Visual analytics** - Interactive charts showing skill distribution and score ranges
- **Export capabilities** - Download all results to Excel for further analysis
- **Customizable weights** - Adjust the importance of different matching factors

---

## Technology Stack

- **Backend**: Python with Flask  
- **NLP Libraries**: spaCy, BERT (Hugging Face Transformers)  
- **Document Processing**: PyPDF2, pdfplumber, python-docx  
- **Frontend**: Bootstrap 5, Chart.js  
- **Data Export**: SheetJS (xlsx)

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sdarmaraj83/matchmakerAI.git
cd matchmakerAI
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 4: Download NLP Models

**spaCy English Model:**

```bash
python -m spacy download en_core_web_sm
```

**BERT Model:**  
This will be automatically downloaded on first run. It uses `bert-base-uncased` from Hugging Face (~440MB).

---

### Step 5: Set Up Required Folders

Ensure the following structure exists:

```text
matchmakerAI/
├── app_rc_M.py          # Main application file
├── requirements.txt     # Python dependencies
├── static/              # Static files (CSS, JS, images)
│   ├── logo.jpeg
│   └── ui_screenshot.png
├── templates/           # HTML templates
│   └── index.html
├── logs/                # Application logs
├── uploads/             # Temporary file uploads (auto-created)
└── README.md            # This documentation
```

To create necessary folders manually:

```bash
mkdir -p static templates logs uploads
```

---

## Running the Application

```bash
python app_rc_M.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser to access the UI.

---

## Usage

1. Enter a job title and description in the left panel.
2. Upload one or more resumes (PDF or DOCX).
3. Adjust matching controls:
   - Similarity Threshold
   - Keyword Boost Weight
   - Exact Match Weight
4. Click **Process Resumes**.
5. View:
   - Match scores
   - Top keywords
   - Skills distribution
   - Score histogram
6. Export data to Excel.

---

## Contributing

We welcome contributions to Matchmaker AI!

### Reporting Issues

Please open an issue on GitHub for bugs or suggestions.

### Making Changes

1. Fork the repository
2. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

3. Make changes and commit:

```bash
git commit -m "Add your feature"
```

4. Push to your fork:

```bash
git push origin feature/your-feature-name
```

5. Create a pull request.

### Pull Request Guidelines

- Follow PEP 8
- Add tests for new features
- Update documentation if needed
- Keep commits focused

---

## Known Limitations

- Large PDFs (>10MB) may be slower
- Image-based PDFs are not supported
- BERT truncates long inputs at 512 tokens

---

## License

Matchmaker AI is released under the [MIT License](LICENSE).

---

## Contact

For questions or support, please contact the maintainer at **s.darmaraj83@gmail.com**.
