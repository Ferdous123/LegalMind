"""Helper to create a minimal valid PDF for integration tests using only stdlib."""


def create_minimal_pdf(path: str, text: str = "Sample legal document text.") -> str:
    """Write a tiny but valid PDF to path, return path.

    Builds a PDF with a single page using raw PDF syntax. No external
    dependencies — the structure is a hand-crafted cross-reference table
    with a single text stream. The text is embedded in the content stream
    using the built-in PDF Helvetica font.

    Args:
        path: Filesystem path where the PDF should be written.
        text: ASCII text string to embed on the single page.

    Returns:
        The same path that was passed in (for chaining).
    """
    # Sanitise: PDF content-stream text must be ASCII; strip or replace
    # characters outside the printable ASCII range.
    safe_text = "".join(
        ch if 32 <= ord(ch) < 127 else " " for ch in text
    )
    # Escape special PDF text characters: backslash, parentheses.
    escaped = safe_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Build the content stream: draw text starting at (50, 750) pt from bottom-left.
    content_stream = (
        "BT\n"
        "/F1 12 Tf\n"
        "50 750 Td\n"
        f"({escaped}) Tj\n"
        "ET\n"
    )
    stream_bytes = content_stream.encode("latin-1")
    stream_length = len(stream_bytes)

    # We will write 4 objects:
    #   1 0 obj  — Catalog
    #   2 0 obj  — Pages node
    #   3 0 obj  — Page (A4: 595 x 842 pt)
    #   4 0 obj  — Content stream
    #   5 0 obj  — Font resource (Helvetica)

    objects: list[bytes] = []

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 595 842]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    obj4 = (
        b"4 0 obj\n"
        b"<< /Length " + str(stream_length).encode() + b" >>\n"
        b"stream\n"
        + stream_bytes +
        b"\nendstream\n"
        b"endobj\n"
    )
    obj5 = (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica\n"
        b"   /Encoding /WinAnsiEncoding >>\n"
        b"endobj\n"
    )

    objects = [obj1, obj2, obj3, obj4, obj5]

    # Compute byte offsets for each object (xref table needs them).
    header = b"%PDF-1.4\n"
    offsets: list[int] = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        pos += len(obj)

    # Build cross-reference table.
    xref_offset = pos
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode()]
    xref_lines.append(b"0000000000 65535 f \n")  # free object 0
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n".encode())

    trailer = (
        b"trailer\n"
        b"<< /Size " + str(len(objects) + 1).encode() + b"\n"
        b"   /Root 1 0 R >>\n"
        b"startxref\n"
        + str(xref_offset).encode() + b"\n"
        b"%%EOF\n"
    )

    with open(path, "wb") as fh:
        fh.write(header)
        for obj in objects:
            fh.write(obj)
        for line in xref_lines:
            fh.write(line)
        fh.write(trailer)

    return path
