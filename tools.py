from typing import Optional
import os
import base64
import PyPDF2
import fitz  # PyMuPDF
from pydantic_ai import RunContext

# dependency container passed to the agent
class Deps:
    def __init__(self, send_event):
        self.send_event = send_event # async callback to push JSON to WebSocket

# read pdf uploaded to pdf_cache
async def read_pdf(
    ctx: RunContext[Deps],
    file_path: str,
    page_number: Optional[int] = None
) -> str:
    """
    Extract text from a PDF. If page number is given, also render that page as an image and send it via the dependency
    callback. The image of the page reading will be shown in the window docked to the right of the web page
    """
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "read_pdf",
        "args": {"file_path": file_path, "page_number": page_number}
    })

    if not os.path.exists(file_path):
        err = f"File not found: {file_path}"
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            if page_number is not None:
                if 1 <= page_number <= num_pages:
                    page = reader.pages[page_number - 1]
                    text = page.extract_text() or ""

                    try:
                        doc = fitz.open(file_path)
                        page_img = doc[page_number - 1]
                        pix = page_img.get_pixmap()
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        await ctx.deps.send_event({
                            "type": "pdf_page_image",
                            "page": page_number,
                            "image": img_base64
                        })
                        doc.close()
                    except Exception as img_err:
                        await ctx.deps.send_event({
                            "type": "warning",
                            "content": f"Could not render page {page_number} as image: {img_err}. Please install PyMuPDF with 'pip install PyMuPDF'."
                        })
                else:
                    text = f"Page {page_number} out of range (1–{num_pages})."
            else:
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"\n--- Page {i+1} ---\n"
                    text += page.extract_text() or ""

        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": text[:200] + "…"})
        return text

    except Exception as e:
        err = f"Error reading PDF: {str(e)}"
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

# pseudo experiment tool function, only used to test whether ai agent can automatically pass in the correct parameters
async def do_experiment(
    ctx: RunContext[Deps],
    spin_speed: int,
    spin_acc: int = 1000,
    spin_dur: int = 30000,
    reagent: str = "",
    volume: int = 10
) -> str:
    """
    Tell the platform to conduct a single round of an in-situ spin coating experiment.
    :param spin_speed: spin speed for spin coating, max 6000rpm
    :param spin_acc: acceleration of the spin coater, must be integer and default 1000rpm/s
    :param spin_dur: spin duration for spin coating in ms. Default 30000ms
    :param reagent: Name of the reagent to be used this round.
    :param volume: The volume of the reagent to be dispensed onto substrate, default 10 ul
    :return: Status message
    """
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "do_experiment",
        "args": {
            "spin_speed": spin_speed,
            "spin_acc": spin_acc,
            "spin_dur": spin_dur,
            "reagent": reagent,
            "volume": volume
        }
    })

    msg = (f"✅ Experiment started: {reagent} at {spin_speed} rpm, "
           f"acc {spin_acc} rpm/s, duration {spin_dur} ms, volume {volume} µl.")
    print(msg)

    await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})
    return msg
