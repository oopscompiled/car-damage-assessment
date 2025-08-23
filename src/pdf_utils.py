import os
import uuid
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# def generate_pdf_report(report_date, user_data, damages, llm_report=None, images=None, output_dir="generated_reports"):
#     pdfmetrics.registerFont(TTFont('NotoSans', 'assets/NotoSans-Regular.ttf'))

#     os.makedirs(output_dir, exist_ok=True)
#     filename = f"{output_dir}/report_{uuid.uuid4().hex[:8]}.pdf"

#     doc = SimpleDocTemplate(filename, pagesize=A4)
#     styles = getSampleStyleSheet()

#     # use 1 style to avoid error
#     styles.add(ParagraphStyle(name="CenterTitle", fontName="NotoSans", fontSize=18, alignment=1, spaceAfter=20))

#     styles['Normal'].fontName = 'NotoSans'
#     styles['Heading2'].fontName = 'NotoSans'
#     styles['Heading2'].fontSize = 14
#     styles['Heading2'].spaceAfter = 12


#     flow = []
#     # LOGO  
#     try:
#         logo = RLImage("assets/logo.png", width=100, height=50)
#         flow.append(logo)
#         flow.append(Spacer(1, 12))
#     except Exception:
#         pass

#     flow.append(Paragraph("Vehicle Damage Assessment Report", styles["CenterTitle"]))
#     flow.append(Paragraph(f"Report Date: {report_date}", styles["Normal"]))
#     flow.append(Spacer(1, 12))

#     # table creation
#     car_data = [
#         ['Name', user_data.first_name],
#         ['Surname', user_data.last_name],
#         ["Make", user_data.make],
#         ["Model", user_data.model],
#         ["Year", user_data.year],
#         ["VIN", user_data.vin]
#     ]
#     table = Table(car_data, colWidths=[80, 350])
#     table.setStyle(TableStyle([
#         ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
#         ("BOX", (0, 0), (-1, -1), 1, colors.black),
#         ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
#         ("ALIGN", (0, 0), (-1, -1), "LEFT"),
#     ]))
#     flow.append(table)
#     flow.append(Spacer(1, 20))

#     # LLM 
#     if llm_report:
#         import markdown as md
#         import re

#         def clean_html_for_paragraph(html):
#             html = re.sub(r"<h\d[^>]*>", "<b>", html)
#             html = re.sub(r"</h\d>", "</b><br/>", html)
#             html = re.sub(r"<ul[^>]*>", "", html)
#             html = re.sub(r"</ul>", "", html)
#             html = re.sub(r"<li>", "• ", html)
#             html = re.sub(r"</li>", "<br/>", html)
#             html = re.sub(r"<p[^>]*>", "", html)
#             html = re.sub(r"</p>", "<br/>", html)
#             return html

#         flow.append(Paragraph("<b>AI-Generated Detailed Report:</b>", styles["Heading2"]))
#         html_report = md.markdown(llm_report)
#         safe_html = clean_html_for_paragraph(html_report)
#         flow.append(Paragraph(safe_html, styles["Normal"]))

#     elif damages:
#         flow.append(Paragraph("<b>Detected Damages:</b>", styles["Heading2"]))
#         damage_data = [[i+1, d] for i, d in enumerate(damages)]
#         damage_table = Table([["#", "Description"]] + damage_data, colWidths=[30, 400])
#         damage_table.setStyle(TableStyle([
#             ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
#             ("BOX", (0, 0), (-1, -1), 1, colors.black),
#             ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
#             ("ALIGN", (0, 0), (0, -1), "CENTER"),
#         ]))
#         flow.append(damage_table)

#     # CAR DAMAGE AND SEGMENTATION IMAGES
#     if images:
#         flow.append(Spacer(1, 20))
#         flow.append(Paragraph("Annotated Images:", styles["Heading2"]))
#         flow.append(Spacer(1, 12))
#         for img_path in images:
#             if os.path.exists(img_path):
#                 try:
#                     flow.append(RLImage(img_path, width=6*inch, height=4*inch))
#                     flow.append(Spacer(1, 12))
#                 except Exception as e:
#                     print(f"Error creating PDF file {img_path}: {e}")
#             else:
#                 print(f"Image not found: {img_path}")

#     # SIGNATURE
#     flow.append(Spacer(1, 40))
#     try:
#         signature = RLImage("assets/signature.png", width=120, height=50)
#         flow.append(signature)
#         flow.append(Paragraph("Inspector Signature", styles["Normal"]))
#     except Exception:
#         pass

#     doc.build(flow)
#     return filename

def generate_pdf_report(report_date, user_data, damages, llm_report=None, images=None, output_dir="generated_reports"):
    import markdown as md
    from bs4 import BeautifulSoup

    pdfmetrics.registerFont(TTFont('NotoSans', 'assets/NotoSans-Regular.ttf'))

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/report_{uuid.uuid4().hex[:8]}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name="CenterTitle", fontName="NotoSans", fontSize=18, alignment=1, spaceAfter=20))
    styles['Normal'].fontName = 'NotoSans'
    styles['Heading2'].fontName = 'NotoSans'
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 12

    flow = []
    # LOGO  
    try:
        logo = RLImage("assets/logo.png", width=100, height=50)
        flow.append(logo)
        flow.append(Spacer(1, 12))
    except Exception:
        pass

    flow.append(Paragraph("Vehicle Damage Assessment Report", styles["CenterTitle"]))
    flow.append(Paragraph(f"Report Date: {report_date}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    # table with car info
    car_data = [
        ['Name', user_data.first_name],
        ['Surname', user_data.last_name],
        ["Make", user_data.make],
        ["Model", user_data.model],
        ["Year", user_data.year],
        ["VIN", user_data.vin]
    ]
    table = Table(car_data, colWidths=[80, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 20))

    # AI-REPORT / DAMAGES
    if llm_report:
        flow.append(Paragraph("<b>AI-Generated Detailed Report:</b>", styles["Heading2"]))

        html_report = md.markdown(llm_report)
        soup = BeautifulSoup(html_report, "html.parser")

        for elem in soup.children:
            if elem.name is None:
                continue

            if elem.name.startswith("h"):
                flow.append(Paragraph(f"<b>{elem.get_text()}</b>", styles["Heading2"]))
                flow.append(Spacer(1, 6))

            elif elem.name == "p":
                text = elem.get_text().strip()
                if text:
                    flow.append(Paragraph(text, styles["Normal"]))
                    flow.append(Spacer(1, 6))

            elif elem.name == "ul":
                for li in elem.find_all("li"):
                    flow.append(Paragraph("• " + li.get_text(), styles["Normal"]))
                flow.append(Spacer(1, 6))

        flow.append(Spacer(1, 12))

    elif damages:
        flow.append(Paragraph("<b>Detected Damages:</b>", styles["Heading2"]))
        damage_data = [[i+1, d] for i, d in enumerate(damages)]
        damage_table = Table([["#", "Description"]] + damage_data, colWidths=[30, 400])
        damage_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ]))
        flow.append(damage_table)

    if images:
        flow.append(Spacer(1, 20))
        flow.append(Paragraph("Annotated Images:", styles["Heading2"]))
        flow.append(Spacer(1, 12))
        for img_path in images:
            if os.path.exists(img_path):
                try:
                    flow.append(RLImage(img_path, width=6*inch, height=4*inch))
                    flow.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Error creating PDF file {img_path}: {e}")
            else:
                print(f"Image not found: {img_path}")

    # SIGNATURE 
    flow.append(Spacer(1, 40))
    try:
        signature = RLImage("assets/signature.png", width=120, height=50)
        flow.append(signature)
        flow.append(Paragraph("Inspector Signature", styles["Normal"]))
    except Exception:
        pass

    doc.build(flow)
    return filename