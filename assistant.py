import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import io
from time import strftime

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from dotenv import load_dotenv
import tools

def read_excel(
        path: str = "./results/PL.xlsx",
        sheet_name: str = 'Sheet1'
) -> str:
    """
    Gets the PL data from Excel. You need to analyze the data first before generating experiment report
    :param path: The path of the data
    :return: Description of the in-situ data
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    time = np.array(df['time/s'].values)
    wavelengths = np.array([i[:2] for i in df.columns[1:]], dtype=np.float64)
    intensity = np.array(df.iloc[:, 1:].values)

    data_description = (
        f"time list: [{time}], wavelengths: [{wavelengths}], intensity matrix:[{intensity}]. "
        "The intensity matrix has rows corresponding to time and columns corresponding to wavelength"
    )
    return data_description

def generate_report(
        analysis: str,
        path: str = "./results/PL.xlsx"
) -> None:
    """
    Generate an experiment report with 3D PL data visualization and analysis text.
    :param analysis: Text analysis to include on the second page
    :param path: Path to the Excel data file
    :return: None
    """
    df = pd.read_excel(path, sheet_name='Sheet1')

    time = np.array(df['time/s'].values)
    wavelengths = np.array([i[:2] for i in df.columns[1:]], dtype=np.float64)
    intensity = np.array(df.iloc[:, 1:].values)

    step = max(1, len(time) // 300)
    X, Y = np.meshgrid(wavelengths[::step], time[::step])
    Z = intensity[::step, ::step]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Intensity (a.u.)', fontsize=11)

    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_zlabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('In-situ PL Data', fontsize=14)
    ax.view_init(elev=25, azim=-60)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()

    # Create PDF
    pdf_filename = f'reports/report{strftime("%y%m%d%H%M%S")}.pdf'
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 50), "Experiment Report", fontsize=14)
    img_data = buf.getvalue()
    rect = fitz.Rect(50, 80, 545, 750)
    page1.insert_image(rect, stream=img_data)
    info_text = f"Time range: {time.min():.1f} - {time.max():.1f} s\nWavelength range: {wavelengths.min():.0f} - {wavelengths.max():.0f} nm"
    page1.insert_text((50, 790), info_text, fontsize=9)

    # Page 2
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 50), "AI Agent Analysis", fontsize=14)
    text_rect = fitz.Rect(50, 80, 545, 800)
    page2.insert_textbox(text_rect, analysis, fontsize=11)

    # Save PDF
    doc.save(pdf_filename)
    doc.close()
    print(f"Report saved as: {pdf_filename}")


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Assistant global history save
history = []
# agent setup
model = OpenAIChatModel(
    "deepseek-reasoner",
    provider=DeepSeekProvider(api_key=api_key),
)
agent = Agent(
    model,
    system_prompt=(
        "You are an lab assistant for an autonomous experiment platform. "
        "Your job is to gather the output data of the experiments conducted in this platform. "
        "You can analyze spectrum data and write experiment records. "
        "After your work is done, please write a short report for your lab leader. "
    ),
    tools=[generate_report, read_excel],
)


async def call_assistant(
    ctx: RunContext[tools.Deps],
    request: str
) -> str:
    """
    This assistant can help with after experiment data analysis and writing experiment reports. It has capable knowledge
    and can give some useful advice for the next round. If the experiment has just finished, it's pleasant to call this
    guy for feedback
    :param request: Tell the assistant what to do at the moment
    :return: A response from the assistant. You can decide your next step according to the feedback
    """
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "call_assistant",
        "args": {"request": request}
    })

    try:
        result = await agent.run(request, message_history=history)
        history.append(result.all_messages())

        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result":"Reporting work..."})

        return "work done"
    except Exception as e:
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result":str(e)})
        return str(e)