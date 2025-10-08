
import fitz  # PyMuPDF
import yaml
import os
import json
import logging
from collections import defaultdict
from PIL import Image, ImageEnhance
import pytesseract
from rapidfuzz import fuzz
import google.generativeai as genai

# ---------------- CONFIG ----------------
GEMINI_API_KEY = "AIzaSyBTjC6ptrYbiZ_glKf-hSBTjRS3RvawZxw"
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ---------------- LOGGER ----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "process.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- HELPERS ----------------

def load_doc_type_rules(yml_path):
    """Load classification rules from YAML and show mappings in CMD."""
    if not os.path.exists(yml_path):
        logger.error(f"Rules file not found: {yml_path}")
        return None

    with open(yml_path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    doc_types = rules.get("doc_types", None)
    if not doc_types:
        logger.error("No 'doc_types' found in YAML.")
        return None

    # 🔑 Show mapping
    print("\n🔑 Mapping of Document Types and Their Keywords:\n")
    for doc_type, config in doc_types.items():
        keywords = config.get("match_keywords", [])
        threshold = config.get("threshold", 60)
        print(f"  • {doc_type:<25} → {keywords} (Threshold: {threshold}%)")
    print("\n" + "=" * 90 + "\n")

    return doc_types


def extract_text_with_ocr(page, page_number):
    """Extract text using PyMuPDF, fallback to Tesseract OCR."""
    text = page.get_text("text").strip()
    if text:
        logger.info(f"[Page {page_number}] Direct text extraction succeeded ({len(text)} chars).")
        return text.lower(), False

    logger.info(f"[Page {page_number}] No text found — using OCR...")
    pix = page.get_pixmap(dpi=400)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    threshold = 110
    img = img.point(lambda x: 0 if x < threshold else 255, '1')

    try:
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        logger.info(f"[Page {page_number}] OCR extracted {len(text)} characters.")
        return text.lower().strip() if text.strip() else "[empty]", True
    except Exception as e:
        logger.exception(f"[Page {page_number}] OCR failed: {e}")
        return "[ocr_failed]", True


def compute_match_ratio(text, keywords):
    matched = []
    for kw in keywords:
        score = fuzz.partial_ratio(kw.lower(), text.lower())
        if score >= 90:
            matched.append(kw)
    ratio = (len(matched) / len(keywords) * 100) if keywords else 0
    return ratio, matched


def classify_page(text, rules):
    best_doc = None
    best_ratio = 0
    best_keywords = []
    for doc_type, props in rules.items():
        if "match_keywords" not in props:
            continue
        keywords = props["match_keywords"]
        threshold = props.get("threshold", 60)
        ratio, matched = compute_match_ratio(text, keywords)
        if ratio > best_ratio:
            best_ratio = ratio
            best_doc = doc_type
            best_keywords = matched
    return best_doc, best_ratio, best_keywords


def gemini_fallback(text, page_number):
    """Ask Gemini LLM to classify unknown pages."""
    prompt = f"""
    You are a document classifier.
    Decide which document type this page is from:
    [W2Form, Paystub, DriverLicense, SocialSecurityAwardLetter,
    PurchaseContractAddendum, FloodCertificate, HazardInsuranceBinder,
    BorrowerAuthorization, EMD, GIFT, TaxCertificate, Other].
    
    
    Return ONLY the document type name.
    

    PAGE TEXT:
    {text[:3500]}
    """
    try:
        response = MODEL.generate_content(prompt)
        doc_type = response.text.strip()
        logger.info(f"[Gemini] Classified page {page_number} as: {doc_type}")
        return doc_type
    except Exception as e:
        logger.error(f"[Gemini] Classification failed for page {page_number}: {e}")
        return "Unclassified"


# ---------------- MAIN FUNCTION ----------------
def split_pdf_by_doc_type(pdf_path, yml_path, output_dir, debug=False):
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)

    SINGLE_PAGE_CLASSES = ["W2Form", "DriverLicense","FloodCertificate"]
    other_pages = []  # 🆕 to collect Gemini single-page docs

    rules = load_doc_type_rules(yml_path)
    if not rules:
        logger.error("No valid rules found.")
        return

    pdf = fitz.open(pdf_path)
    logger.info(f"📘 Loaded PDF: {pdf_path} ({len(pdf)} pages)")

    grouped_pages = defaultdict(list)
    unclassified_pages = []
    summary = []

    for i, page in enumerate(pdf):
        page_num = i + 1
        logger.info(f"\n===== Processing Page {page_num} =====")
        text, used_ocr = extract_text_with_ocr(page, page_num)

        text_file = os.path.join(text_dir, f"page_{page_num}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

        doc_type, ratio, matched_keywords = classify_page(text, rules)
        threshold = rules.get(doc_type, {}).get("threshold", 60) if doc_type else 60

        if doc_type and ratio >= threshold:
            if doc_type in SINGLE_PAGE_CLASSES:
                # ✅ Save single-page docs immediately
                doc_folder = os.path.join(output_dir, doc_type)
                os.makedirs(doc_folder, exist_ok=True)
                new_pdf = fitz.open()
                new_pdf.insert_pdf(pdf, from_page=i, to_page=i)
                out_path = os.path.join(doc_folder, f"{doc_type}_page_{page_num}.pdf")
                new_pdf.save(out_path)
                new_pdf.close()
                logger.info(f"[Page {page_num}] 📄 Saved single-page {doc_type}: {out_path}")
            else:
                grouped_pages[doc_type].append(i)
                logger.info(f"[Page {page_num}] ✅ Classified as: {doc_type} ({ratio:.1f}% ≥ {threshold}%)")
        else:
            logger.warning(f"[Page {page_num}] ⚠️ Unclassified (best: {doc_type or 'None'} {ratio:.1f}% < {threshold}%)")
            unclassified_pages.append((i, text))

        summary.append({
            "page": page_num,
            "type": doc_type if doc_type else "Unclassified",
            "match_ratio": ratio,
            "matched_keywords": matched_keywords,
            "ocr_used": used_ocr,
            "source" : "RapidFuzz"
        })

    # ---------------- GEMINI FALLBACK ----------------
    if unclassified_pages:
        logger.info("\n🤖 Running Gemini fallback for unclassified pages...")
        for idx, text in unclassified_pages:
            doc_type = gemini_fallback(text, idx + 1)

            if doc_type in SINGLE_PAGE_CLASSES:
                logger.info(f"[Gemini Fallback] Page {idx+1} is {doc_type} → will be added to Other.pdf")
                other_pages.append(idx)  # 🆕 collect for Other.pdf
            elif doc_type in ["Other", "Unclassified"]:
                other_pages.append(idx)  # 🆕 also go to Other.pdf
            else:
                grouped_pages[doc_type].append(idx)

            summary[idx]["type"] = doc_type
            summary[idx]["source"] = "Gemini"
    
    print(grouped_pages)

    # ---------------- SAVE OUTPUT FILES ----------------
    if other_pages:
        other_pdf = fitz.open()
        for p in sorted(set(other_pages)):
            other_pdf.insert_pdf(pdf, from_page=p, to_page=p)

        out_path = os.path.join(output_dir, "Other.pdf")  # ✅ Directly under output_docs/
        other_pdf.save(out_path)
        other_pdf.close()
        logger.info(f"📂 Saved → {out_path} ({len(other_pages)} pages total)")

# 🧩 Now handle grouped (non-single) pages — preserving original order
    for doc_type, pages in grouped_pages.items():
        if not pages:
            continue

    # 🔹 Special case: skip duplicate "Other" — we already saved it
        if doc_type == "Other":
            continue

    # Normal document type PDFs
        doc_folder = os.path.join(output_dir, doc_type)
        os.makedirs(doc_folder, exist_ok=True)

    # ✅ Preserve original order of pages (sorted by position in the PDF)
        new_pdf = fitz.open()
        for p in sorted(pages):
            new_pdf.insert_pdf(pdf, from_page=p, to_page=p)

        out_path = os.path.join(doc_folder, f"{doc_type}.pdf")
        new_pdf.save(out_path)
        new_pdf.close()
        logger.info(f"📂 Saved → {out_path} ({len(pages)} pages in original order)")

    # ---------------- SUMMARY ----------------
    summary_path = os.path.join(output_dir, "classification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n📊 Summary saved → {summary_path}")

    # total_classified = sum(len(v) for v in grouped_pages.values()) + len(other_pages)
    # logger.info(f"\n🧾 Final Report: {total_classified}/{len(pdf)} pages classified successfully.")
    pdf.close()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    split_pdf_by_doc_type(
        pdf_path=r"C:\Users\PawanMagapalli\Downloads\document\Doc-Classification\merged doc.pdf",
        yml_path=r"C:\Users\PawanMagapalli\Downloads\new_approach\new_approach\rule.yaml",
        output_dir=r"output_docs",
        debug=True
    )

