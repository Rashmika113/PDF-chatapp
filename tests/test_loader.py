from rag.loader import load_pdf


def test_load_pdf_extracts_each_page(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.page_number for p in pages] == [1, 2]
    assert pages[0].source == "handbook.pdf"
    assert "25 days" in pages[0].text
    assert "password" in pages[1].text


def test_load_pdf_from_file_object(sample_pdf):
    with open(sample_pdf, "rb") as f:
        pages = load_pdf(f, source="upload.pdf")
    assert pages[0].source == "upload.pdf"
