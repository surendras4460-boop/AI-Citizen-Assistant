"""Generate SYNTHETIC test documents (fictional person, no real records) and ground truth."""
import json, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from PIL import Image, ImageDraw

ss = getSampleStyleSheet(); H, B = ss["Heading2"], ss["BodyText"]
FOOT = "SYNTHETIC TEST DOCUMENT - fictional data, not a real record."

def build(path, pages):
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    story = []
    for i, pg in enumerate(pages):
        for kind, text in pg:
            story.append(Paragraph(text, H if kind == "h" else B)); story.append(Spacer(1, 9))
        story.append(Paragraph(FOOT, B))
        if i < len(pages) - 1: story.append(PageBreak())
    doc.build(story)

D = "docs/"
build(D+"resume_asha_verma.pdf", [
 [("h","ASHA VERMA - Curriculum Vitae"),
  ("p","Email: asha.verma.test@example.com | Phone: +91-90000-00000 | Address: Rampur Road, Riverside District."),
  ("p","<b>Career Objective:</b> Final-year engineering student seeking an entry-level data analyst role where statistical thinking and Python skills can be applied to real business problems."),
  ("p","<b>Education:</b> B.Tech in Computer Science and Engineering, Northfield Institute of Technology (2023-2027), CGPA 8.4. Senior Secondary (Class 12), Lakeview Public School, 91.8 percent. Secondary (Class 10), Lakeview Public School, 93.2 percent."),
  ("p","<b>Technical Skills:</b> Python, SQL, Pandas, NumPy, scikit-learn, Power BI, Tableau, Git and Streamlit. Soft skills include communication, teamwork and time management."),
  ("p","<b>Languages Known:</b> English and Hindi.")],
 [("h","Projects, Certifications and Achievements"),
  ("p","<b>Projects:</b> (1) Retail Sales Dashboard - an interactive Power BI dashboard built on 50,000 transaction records. (2) Student Dropout Predictor - a random forest classifier reaching 87 percent accuracy on a public education dataset. (3) Campus Event Chatbot - a Streamlit chatbot that answers timetable queries."),
  ("p","<b>Certifications:</b> Google Data Analytics Professional Certificate (2025); Microsoft Azure Fundamentals AZ-900 (2025); SQL Advanced, HackerRank (2024)."),
  ("p","<b>Achievements:</b> Winner of the Inter-College Hackathon 2025, first place among 42 teams; Dean's merit list for four consecutive semesters; Coordinator of the Data Science Club.")]])

build(D+"marksheet_class10.pdf", [[("h","Secondary School Examination - Statement of Marks (Sample Board)"),
  ("p","Name of Candidate: Asha Verma. Roll Number: 1234567. Year of Examination: 2021. School: Lakeview Public School."),
  ("p","Subject-wise marks out of 100: English 92, Hindi 95, Mathematics 96, Science 93, Social Science 90."),
  ("p","Total marks obtained: 466 out of 500. Percentage: 93.2. Result: PASS with First Division.")]])

build(D+"marksheet_class12.pdf", [[("h","Senior School Certificate Examination - Statement of Marks (Sample Board)"),
  ("p","Name of Candidate: Asha Verma. Roll Number: 7654321. Year of Examination: 2023. Stream: Science with Computer Science."),
  ("p","Subject-wise marks out of 100: English 88, Physics 90, Chemistry 92, Mathematics 94, Computer Science 95."),
  ("p","Total marks obtained: 459 out of 500. Percentage: 91.8. Result: PASS with Distinction.")]])

build(D+"btech_gradecard.pdf", [
 [("h","Northfield Institute of Technology - Consolidated Grade Card (Semesters 1-4)"),
  ("p","Student: Asha Verma. Programme: B.Tech Computer Science and Engineering. Admission year: 2023."),
  ("p","Semester 1 SGPA 8.1. Semester 2 SGPA 8.3. Semester 3 SGPA 8.6 (Data Structures A, Discrete Mathematics A, Digital Logic B+, Physics B+). Semester 4 SGPA 8.5 (Algorithms A, Database Systems A, Operating Systems B+, Probability B+).")],
 [("h","Consolidated Grade Card (Semesters 5-6)"),
  ("p","Semester 5 SGPA 8.2: Machine Learning A, Computer Networks B, Software Engineering A, Data Mining B+. The weakest grade in semester 5 was in Computer Networks (B)."),
  ("p","Semester 6 SGPA 8.7: Deep Learning A, Cloud Computing A, Data Visualisation A, Information Security B+."),
  ("p","Cumulative Grade Point Average (CGPA) up to semester 6: 8.4.")]])

build(D+"income_certificate.pdf", [[("h","Income Certificate (Sample Tehsil Office)"),
  ("p","Certificate No. IC/2026/00421. Issued on 12 April 2026 by the Tehsildar, Sample Tehsil, Riverside District."),
  ("p","This is to certify that Shri Rakesh Verma, father of Asha Verma, has an annual family income from all sources of Rs. 1,85,000 (Rupees One Lakh Eighty-Five Thousand only) for the financial year 2025-26."),
  ("p","This certificate is valid for one year from the date of issue.")]])

build(D+"domicile_certificate.pdf", [[("h","Domicile / Residence Certificate (Sample District Office)"),
  ("p","Certificate No. DC/2026/01877. Issued on 20 March 2026."),
  ("p","Certified that Asha Verma, daughter of Rakesh Verma, is a permanent resident of Village Rampur, Block Sadar, Riverside District, Sample State, and has been residing there for more than 15 years.")]])

# image-only (scanned-style) PDF to test the skip behaviour
img = Image.new("RGB", (900, 400), "white"); d = ImageDraw.Draw(img)
d.text((30, 60), "Scanned caste certificate page (image only)", fill="black")
img.save("scan.png")
from reportlab.pdfgen import canvas
c = canvas.Canvas(D+"scanned_only.pdf", pagesize=A4); c.drawImage("scan.png", 50, 500, width=450, height=200); c.showPage(); c.save()

R, I, C10, C12, BT, INC, DOM = ("resume_asha_verma.pdf","income_certificate.pdf","marksheet_class10.pdf",
    "marksheet_class12.pdf","btech_gradecard.pdf","income_certificate.pdf","domicile_certificate.pdf")
answerable = [
 ("Which institute is the applicant pursuing her engineering degree from?", [(R,1)]),
 ("List the programming languages and analytics tools she is skilled in.", [(R,1)]),
 ("How can the candidate be contacted by email?", [(R,1)]),
 ("What projects has the candidate worked on?", [(R,2)]),
 ("Does she hold any cloud computing certification?", [(R,2)]),
 ("Has she won any competition?", [(R,2)]),
 ("How many marks did she get in Mathematics in her secondary school examination?", [(C10,1)]),
 ("What percentage did she secure in the tenth grade?", [(C10,1),(R,1)]),
 ("What was her roll number for the Class 10 examination?", [(C10,1)]),
 ("Which stream did she study in senior secondary school?", [(C12,1)]),
 ("What was her roll number in the twelfth standard exam?", [(C12,1)]),
 ("How many marks did she score in Physics in Class 12?", [(C12,1)]),
 ("What SGPA did she obtain in the third semester?", [(BT,1)]),
 ("What is her cumulative grade point average after the sixth semester?", [(BT,2),(R,1)]),
 ("Which subject got the weakest grade in semester 5?", [(BT,2)]),
 ("What is the certified yearly income of the family?", [(INC,1)]),
 ("Until when is the income certificate valid?", [(INC,1)]),
 ("For how many years has she been living in her village?", [(DOM,1)]),
 ("What is the certificate number of the residence proof?", [(DOM,1)]),
 ("What is the name of the applicant's father?", [(INC,1),(DOM,1)]),
]
absent = ["What is the applicant's Aadhaar number?","What is her PAN card number?","What is her bank account number?",
 "Which caste category does she belong to?","What is her blood group?","What is her passport number?",
 "Does she have a driving licence?","What is the name of her spouse?","What is her ration card number?","What is her monthly stipend?"]
json.dump({"answerable":[{"q":q,"truth":[list(t) for t in tr]} for q,tr in answerable],"absent":absent}, open("ground_truth.json","w"), indent=1, ensure_ascii=False)
print("docs written; answerable",len(answerable),"absent",len(absent))
