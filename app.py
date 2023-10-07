from pathlib import Path

import streamlit as st
from PIL import Image


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "CV_Ajeet_Kumar_Verma.pdf"
profile_pic = current_dir / "assets" / "aj.png"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Ajeet"
PAGE_ICON = ":wave:"
NAME = "Ajeet Kumar Verma"
DESCRIPTION = """
Senior AI Engineer/ Data Scientist
"""
EMAIL = "ajeet214@outlook.com"
CONTACTS = "+84 384296608"
LOCATION = "Hanoi, Vietnam"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/ajeet214/",
    "GitHub": "https://github.com/ajeet214",
    "Twitter": "https://twitter.com/ajeet214",
    "Medium": "https://medium.com/@ajeet214",
    "StackOverflow": "https://stackoverflow.com/users/11179336/ajeet-verma"
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.write(":email:", EMAIL)
    st.write(":iphone:", CONTACTS)
    st.write(":house:", LOCATION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

st.write("---")

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Professional Summary:")
st.write(
    """
- An Indian expatriate, currently living and working in Vietnam. skilled as an AI engineer and data scientist with over 7 years of professional experience in the IT industry, I bring a strong foundation of technical expertise and business understanding to an organization. My skills in project management and business analysis, combined with my technical knowledge of AI and data science, enable me to identify and solve complex problems, develop innovative solutions, and drive business growth and efficiency. I can communicate effectively with both technical and non-technical stakeholders and have a track record of building strong relationships and delivering successful outcomes
- Experienced AI/ML Engineer with a demonstrated history of working in the information technology and services industry. Skilled in Artificial Intelligence and Machine learning, deep learning, Python (Programming Language), Web Scraping, etc
- Performed several roles as AI/ML Engineer, Data Engineer, Python developer, and Senior Software Engineer.
- Have consistently contributed to the organization’s growth and profitability by combining strong technical, and management knowledge and a dedicated proactive approach.
- Cooperative and able to perform within a team-oriented atmosphere.
- Experience in integrating applications with third-party web services & APIs.
- Experience in building ETL, NLP, CV, and ML pipelines.
- Knowledge of predictive/ prescriptive analytics including ML algorithms.
- Experience in scrum/agile framework.
- Experience in container tools such as Docker.
- Experience in processing various forms of raw data such as videos, images, and texts.
- Experience in developing microservice architecture and API Frameworks supporting application development.
- Strong verbal and written communication skills.
"""
)

# --- EDUCATIONAL BACKGROUND ---
st.write('\n')
st.subheader("Educational Background:")
st.write(
    """
- Post Graduate Diploma in Machine Learning and Artificial Intelligence, IIIT Bangalore, India.
- Bachelor of Technology in Electrical Engineering (Hons), Rajasthan Technical University, Kota, India.
"""
)

# --- TRAINING & CERTIFICATIONS: ---
st.write('\n')
st.subheader("Training & Certifications:")
st.write(
    """
- Generative AI with Large Language Models by Coursera
- Generative Adversarial Networks (GANs) by Coursera
- Data Scientist Nanodegree by Udacity
- Computer Vision Nanodegree by Udacity
- Microsoft Certified Azure AI Engineer Associate
- Professional Certification in Project Management by Google
- TensorFlow Developer Professional Certification by DeepLearning.AI
- Data Science: Visualization by HarvardX
- Data Science: R basics by HarvardX
- Machine Learning Data Lifecycle in Production by Coursera
- Introduction to Machine Learning in Production by Coursera
- Data Analysis Using Pyspark by Coursera
- Specialized Models Time Series and Survival by Coursera
- AWS S3 Basics by Coursera
- Introduction to Big Data with Spark and Hadoop by IBM
- Data Engineering and Machine Learning using Spark by IBM
- Data Science Methodologies by IBM
- Data Science Tools by IBM
- Docker Essentials by IBM
- Hands-On Introduction to Apache Airflow by Udemy
- Style Transfer using GANs
- Self-Awareness and the Effective Leader by Rice University
- Version Control with Git by Atlassian
- Agile with Atlassian Jira
- ETS TOEIC®
"""
)


# --- TECHNICAL EXPERTISE & SKILLS ---
st.write('\n')
st.subheader("Technical Expertise & Skills:")
st.write(
    """
- Programming: Python, R
- Database: MySQL, MongoDB
- Web scrapping: Scrapy, Selenium, Beautiful Soup, Requests, Lxml, etc.
- Machine learning/deep learning framework: Scikit-learn, TensorFlow, Keras, PyTorch
- Version control tools: Git, Gitlab, Bitbucket
- Data analytics and processing tools: NumPy, SciPy, Pandas, Polars, Pyspark 
- Data Visualization tools: Matplotlib, Seaborn, Plotly, Kibana
- Image processing tools: OpenCV, PIL, Scikit-Image
- DevOps tools: Docker, Terraform 
- Restful API, FastAPI, Flask, Regex, Apache Spark, Apache Airflow, AWS, Elastic Stack (EFK).
- Computer Vision, Recommendation System, NLP, OCR, and Time-Series Analysis (Statsmodels).
- Generative AI, Large Language Models, LangChain, GANs.
"""
)

# --- LANGUAGE SKILL ---
st.write('\n')
st.subheader("Language Skills:")
st.write(
    """
- Fluent in English
- Native in Hindi
- Elementary in Korean
- Elementary in Vietnamese
"""
)


# --- PROFESSIONAL EXPERIENCE ---
st.write('\n')
st.subheader("Professional Experience:")
st.write("---")

# --- JOB 1
st.write("🚧", f"**AI Engineer for the Covestro ICON project in [FPT Software](https://fptsoftware.com/), Hanoi, Vietnam.**")
st.write("06/2023 – Present")
st.write("Project Description:")
st.write(
    """
- The primary objective of the ICon project is to establish a robust and user-friendly platform that enables natural language-based access to Covestro's business data, with a specific focus on Sales data sourced from SAP BW. The project aims to:
- Facilitate Natural Language Interaction
- Enable Descriptive Data Reporting
- Enhance Decision-Making
"""
)
st.write("Team size: 09")
st.write("Responsibilities:")
st.write("""
- Undertake extensive research to identify the most appropriate AI algorithms and tools for the project, implement them effectively, and optimize their performance to achieve the desired outcomes.
- Understand and capture the key business requirements of the Stakeholder/client.
- Collaborate closely with cross-functional teams, including data engineers, data scientists, software developers, and business stakeholders, to align the AI components with overall project goals and objectives.
- Stay updated with the latest developments in LLMs, and conversational AI technologies. Explore innovative solutions and enhancements to keep the project at the forefront of industry standards.
- Model Selection and Configuration
- Large Language Models Fine-tuning and Training
""")
st.write("""
Technologies used: 
- Python
- Large Language Models
- MySQL
- PyTorch
- AWS	
- LangChain
- Git
""")

# --- JOB 2
st.write("🚧", "**AI Engineer for the AkaCam project in [FPT Software](https://fptsoftware.com/), Quy Nhon, Vietnam.**")
st.write("05/2022 – 06/2023")
st.write("Project Description:")
st.write(
    """
- AkaCam is a project that uses image data to intelligently analyze human behavior and is powered by AI
- Object detection
- Object classification
- Object tracking
- Object re-identification
"""
)
st.write("Team size: 19")
st.write("Responsibilities:")
st.write("""
- Undertake extensive research to identify the most appropriate AI algorithms and tools for the project, implement them effectively, and optimize their performance to achieve the desired outcomes.
- Understand and capture the key business requirements of the Stakeholder/client.
- Conduct a thorough examination of the current code base to gain a comprehensive understanding of the system's architecture, design patterns, and programming practices.
- Enhance the overall system performance by refactoring the AI services, which involves optimizing code, improving algorithms, and addressing any technical debt.
- Tailor the existing packages to meet specific client requirements by implementing new features, customizing existing functionalities, and integrating additional logic as necessary.
- Ensure that the software is efficient, maintainable, and scalable by adhering to software engineering best practices, such as modularization, documentation, and testing.
- Collaborate with cross-functional teams, including product managers, software developers, and quality assurance engineers, to ensure that the AI services meet the product's requirements and specifications.
- Continuously monitor and evaluate the system's performance, identify any bottlenecks or issues, and implement necessary changes to optimize performance and improve the user experience.
""")
st.write("""
Technologies used: 
- Python
- Deep learning algorithms 
- Data processing
- Data Visualization
- Computer Vision
""")

# --- JOB 3
st.write("🚧", "**ML Engineer for the GoToro project in [FPT Software](https://fptsoftware.com/), Quy Nhon, Vietnam.**")
st.write("11/2021 – 06/2022")
st.write("Project Description:")
st.write(
    """
Build an Intelligent Job portal service as a Self-managed for GoToro Clients. The goal is to help high-volume recruitment industries accelerate the hiring processes and increase the efficiency of recruitment campaigns by infusing Analytics and Artificial intelligence.
"""
)
st.write("Team size: 26")
st.write("Responsibilities:")
st.write("""
- Engage with the Stakeholders and clients to understand and capture the business goals and translate business problems into an analytical framework.
- Research and implement appropriate ML algorithms and tools.
- Select appropriate datasets and data representation methods and build the ETL pipeline.
- Collaborate, help, and exchange knowledge with the team members, peers, and leadership at QAI to foster a community-working environment.
- Implement the time-series ML algorithm for bidding forecast and budget allocation to the publishers on the job boards.
""")
st.write("""
Technologies used: 
- Python 
- ML algorithms
- Data processing
- Data visualization
- Git
- MySQL
- MongoDB
- Time-series Analysis
""")

# --- JOB 4
st.write("🚧", "**AI (Computer Vision) Engineer, for Smart City project in [Viettel](https://viettel.com.vn/en/), Hanoi, Vietnam**")
st.write("10/2020 – 10/2021")
st.write("Project Description:")
st.write(
    """
- The project objective revolves around solving the manual inspection problem involved in checking, monitoring, and reporting the quality of various components and pieces of equipment installed/placed at the telecom sub-stations and automating the whole process using AI.
- Developing machine learning and deep learning models to perform visual recognition tasks, such as object classification, segmentation, and detection.
- Optical character recognition
"""
)
st.write("Team size: 14")
st.write("Responsibilities:")
st.write("""
- Part of the AI team majorly focused on computer vision projects.
- Training machine learning and deep learning models to perform visual recognition tasks, such as classification, segmentation, and detection.
- Optimizing deep neural networks and the related preprocessing/post-processing code. 
- Reading research papers, implementation, and its incorporation into the existing ecosystem.
- Responsible for automation of manual inspection tasks such as equipment validation, fault detection, structural abnormalities, and anomaly detection using deep learning.
- Work on computer vision algorithms and image processing, including rule-based image processing techniques and deep learning-based algorithms.
- Utilize annotation tools for accurately annotating images and videos for tasks such as object detection, segmentation, and classification.
""")
st.write("""
Technologies used: 
- Python
- PyTorch 
- Machine Learning, Deep Learning
- Computer Vision 
- Image Processing
- Docker
- OpenCV
- FastAPI
- OCR
- Recommendation System
""")

# --- JOB 5
st.write("🚧", "**AI Engineer, Data Engineer, Smart City project in [Viettel](https://viettel.com.vn/en/), Hanoi, Vietnam**")
st.write("04/2021 – 06/2021")
st.write("Project Description:")
st.write(
    """
- Develop a dashboard for AI service log monitoring and analysis.
- Create the whole ETL pipeline for ingesting, transforming, and loading the logs to the dashboard.
"""
)
st.write("Team size: 14")
st.write("Responsibilities:")
st.write("""
- Part of the AI team majorly focused on computer vision projects.
- Design and develop the EFK architecture.
- Responsible for creating different visualizations to efficiently analyze and monitor the logs to derive business value.
- Implement the role-based access control for the logging stack.
- Responsible for developing and deploying the Pipeline. 
""")
st.write("""
Technologies used: 
- Python
- Ruby 
- Elasticsearch
- Fluentd
- Kibana
- Regex
- Docker
""")

# --- JOB 6
st.write("🚧", "**Senior Software Engineer for Viavi project in [Altran (Capgemini Engineering)](https://www.capgemini.com/about-us/who-we-are/our-brands/capgemini-engineering/), Gurugram, India.**")
st.write("04/2021 – 06/2021")
st.write("Project Description:")
st.write(
    """
Maintain, customize and improve the automation framework for network test and measurement, network assurance, and optical solutions.
"""
)
st.write("Team size: 12")
st.write("Responsibilities:")
st.write("""
- Communicate and understand the business requirements with customers.
- Review the existing code base and refactor or optimize for better performance.
- Develop automation scripts for 4G and 5G network device testing and monitoring.
""")
st.write("""
Technologies used: 
- Python
- Restful API
- Flask
- MySQL
- PyTest
- Git
""")

# --- JOB 7
st.write("🚧", "**Technical Consultant, Data Engineer for Innowatts project in [Accolite](https://www.accolite.com/), Gurugram, India.**")
st.write("09/2019 – 03/2020")
st.write("Project Description:")
st.write(
    """
Innowatts’ SaaS platform transforms how energy providers understand and serve their customers. Using AI and learnings from more than 45 million global meters to help energy retailers, utilities and grid operators unlock meter-level data, understand their customers, and make business processes automated and smarter."""
)
st.write("Team size: 11")
st.write("Responsibilities:")
st.write("""
- Engage with clients to gather requirements, and provide them with solutions that are tailored to their specific needs and that meet their expectations.
- Automating the data pipelines with Airflow.
- Workflow Orchestration using directed acyclic graphs (DAGs) to coordinate tasks and dependencies.
- Responsible for creating maintainable, versioned, testable, and configurable DAGs. 
- Coordinate with internal teams to understand client requirements and provide technical solutions.
- Create logic for Data processing and computations.
- Extracts the data from different RDBMS source systems, then transforms the data (like applying calculations, concatenations, etc.) and finally loads the data into the Data Warehouse system.
""")
st.write("""
Technologies used: 
- Python
- AWS (S3, Athena, EMR)
- Apache Airflow 
- Git
- MySQL
- Pyspark
""")

# --- JOB 8
st.write("🚧", "**Python Developer, Web Scrapper for WebMine project in [SecNinjaz Technologies LLP](https://www.secninjaz.com/), India.**")
st.write("09/2019 – 03/2020")
st.write("Project Description:")
st.write(
    """
WebMine is a revolutionary & state-of-the-art OSINT (Open-source intelligence) based product utilizing AI & ML to produce virtual Intelligence and gather information from diverse sources including search engines, social media platforms, the dark web, etc.
""")
st.write("Team size: 19")
st.write("Responsibilities:")
st.write("""
- Responsible for developing several micro-service architectures.
- Responsible for gathering data from various platforms (social media, search engines, etc.) through either APIs or web scrapping and creating APIs to integrate with the existing micro-services.
- Build reusable code and libraries for future use. 
- Focus on test-driven development and create code reviews.
- Run and monitor performance tests on new and existing software to correct mistakes, isolate areas for improvement, and debug.
- Identify, leverage, and successfully evangelize opportunities to improve engineering productivity.
- Collaboration with team members to establish back-end conventions, principles, and patterns.
- Performs technical analysis for the development of new features.
""")
st.write("""
Technologies used: 
- Python 
- Selenium 
- Docker
- MongoDB, MySQL
- Flask
- Git
- Restful API 
- PyTest
- Scrapy      
- Elasticsearch
- Kibana
- Beautiful Soup, Requests, Urllib3, etc.
- Regex
""")

# --- JOB 9
st.write("🚧", "**Python Developer, ML Trainer for Forsk project in Seedlogix Resources India Pvt. Ltd, Jaipur, India.**")
st.write("06/2015 – 01/2018")
st.write("Project Description:")
st.write(
    """
Forsk with a mission to empower people with top-quality educational programs in the domain of software engineering and information technology and helps university students fill the industry gap by improving skills in coding and computer science fundamentals and secure quality careers in the industry. 
""")
st.write("Team size: 8")
st.write("Responsibilities:")
st.write("""
- Responsible for developing and delivering lectures, workshops, and other learning materials on a variety of topics related to ML, and Python.
- Responsible for designing and implementing assessments and other activities to help students learn and apply their knowledge.
- Involve in creating quizzes, exams, projects, and other assignments, as well as providing feedback and support to students.
- Research, and work on data science and machine learning projects.
- Study and transform data science prototypes.
- Research and implement appropriate ML algorithms and tools.
- Select appropriate datasets and data representation methods.
- Run machine learning tests and experiments.
- Perform statistical analysis and fine-tuning using test results.
- Stay up-to-date on the latest developments in the field, and for adapting the teaching materials and approaches to reflect these developments.
""")
st.write("""
Technologies used: 
- Python
- Machine Learning Algorithms
- TensorFlow
- MySQL
- Flask
- Git
- HTML, CSS
- C, C++
- Restful API 
- Web scrapping.
- OpenCV
- Image processing
- Data processing
""")

