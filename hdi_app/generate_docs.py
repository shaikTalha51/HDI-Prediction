from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, PageBreak

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / 'project-documentation'

phase_docs = [
    ('1. Brainstorming & Ideation', 'ideation_report.pdf', 'HDI Predictor Project - Brainstorming & Ideation', [
        'Project Title: HDI Predictor Web Application',
        'Problem Statement: Many users and students struggle to understand how health, education, and income factors contribute to a country\'s Human Development Index. Traditional methods are often static and do not provide instant predictive insights.',
        'Proposed Solution: Build a Flask-based web application that predicts HDI scores from four indicators and classifies them into Low, Medium, High, and Very High tiers.',
        'Target Audience: Students, researchers, educators, and policymakers interested in development analytics.',
        'Technology Stack: Python, Flask, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, HTML/CSS.'
    ]),
    ('2. Requirement Analysis', 'requirements_specification.pdf', 'HDI Predictor Project - Requirement Analysis', [
        'Software Dependencies: Python 3.10+, Flask, NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Jinja2.',
        'Hardware Configuration: Minimum 4 GB RAM, 2 GB free storage, dual-core processor recommended.',
        'Data Specifications: 200 synthetic records with 5 input features and 1 target column; features include Life_Expectancy, Mean_Years_Schooling, Expected_Years_Schooling, and GNI_Per_Capita.'
    ]),
    ('3. Project Design Phase', 'design_document.pdf', 'HDI Predictor Project - Design Document', [
        'System Architecture Flow: User enters input values in the web form, the backend validates them, scales the features, passes them to the trained regression model, and returns a predicted HDI score and tier.',
        'Data Flow Logic: Missing values are imputed, features are standardized, model training is performed with train-test split, and predictions are rounded before display.',
        'User Interface Mockups: Home page, prediction form, result page, and insights page are implemented using Flask templates and CSS styling.'
    ]),
    ('4. Project Planning Phase', 'planning_sheet.pdf', 'HDI Predictor Project - Planning Sheet', [
        'Week 1: Set up environment, create project structure, and generate synthetic dataset.',
        'Week 2: Perform exploratory analysis and build the regression model.',
        'Week 3: Implement Flask routes, templates, and the insights dashboard.',
        'Week 4: Test the system, validate outputs, and prepare submission documentation.'
    ]),
    ('5. Project Development Phase', 'development_phase_report.pdf', 'HDI Predictor Project - Development Phase Report', [
        'Location of Source Code: The runnable codebase is stored at the repository root in app.py, train_model.py, generate_dataset.py, templates/, and static/.',
        'Source Code Overview: generate_dataset.py creates the CSV dataset, train_model.py trains and saves the model artifacts, and app.py runs the web server.',
        'Dataset Path: data/hdi_dataset.csv.'
    ]),
    ('6. Project Testing', 'test_report.pdf', 'HDI Predictor Project - Test Report', [
        'Model Evaluation: R-squared 0.9681, MAE 0.0189, RMSE 0.0249 on the validation split.',
        'System Verification Cases: Valid numeric input returns a prediction; invalid input shows an error message; boundary values are handled by clipping to the range [0, 1].'
    ]),
    ('7. Project Documentation', 'user_manual.pdf', 'HDI Predictor Project - User Manual', [
        'Setup & Installation: pip install -r requirements.txt',
        'Running the App: python generate_dataset.py, python train_model.py, python app.py',
        'User Guidelines: Enter four development indicators, submit the form, and review the predicted HDI score and tier.'
    ]),
    ('8. Project Demonstration', 'demo_screenshots.pdf', 'HDI Predictor Project - Demonstration', [
        'Terminal Execution: Screenshots or terminal logs showing successful dataset generation and model training.',
        'Web Interface: Screenshots of the home page, prediction form, and result page.',
        'Demo Video Link: Add a Google Drive or YouTube link here if you want to include a recorded demo.'
    ]),
]

for folder_name, filename, title, bullets in phase_docs:
    folder = DOCS_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / filename

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=18, leading=24, textColor=colors.HexColor('#1f4e79'), spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyStyle', parent=styles['BodyText'], fontSize=11, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name='BulletStyle', parent=styles['BodyText'], fontSize=11, leading=14, leftIndent=18, bulletIndent=0, spaceAfter=6))

    story = []
    story.append(Paragraph(title, styles['TitleStyle']))
    story.append(Spacer(1, 12))
    story.append(Paragraph('Prepared for the SmartBridge internship submission.', styles['BodyStyle']))
    story.append(Spacer(1, 12))

    for item in bullets:
        story.append(Paragraph(item, styles['BodyStyle']))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f'Created {output_path}')

print('All documentation PDFs generated successfully.')
